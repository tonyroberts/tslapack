import {description, program, requiredArg, usage, version, optionalArg} from 'commander-ts';
import * as fs from "fs";
import * as path from 'path';
import * as constants from './constants';
import {Parameter} from './Parameter';


@program()
@version('1.0.0')
@description('Code generator for tslapack')
@usage('--help')
export class Program {

    constructor() {}

    run(@requiredArg('lapack-src') lapack: string,
        @requiredArg('output') output: string,
        @optionalArg('function') func: string | null) {
        console.log(`Searching ${lapack} for LAPACK source code`);
        let files = fs.readdirSync(lapack);

        // Only get the double precision functions for now
        files = files.filter(x => x.match('^d.+\.f'));
        let functionsOutputDir = path.join(output, 'functions')
        let generatedFunctions: string[] = []
        for(let file of files) {
            if (func && file.toLowerCase() != func.toLowerCase() + '.f') {
                continue;
            }

            let fullname = path.join(lapack, file);
            try {
                let name = Program.processSubroutine(fullname, functionsOutputDir);
                generatedFunctions.push(name);
            }
            catch (e) {
                console.error(`Error processing ${file}: ${e}`);
            }
        }

        // Write out the main exports file
        Program.writeExportsFile(generatedFunctions, output);
    }

    /*
     * Parses the lines from the FORTRAN code looking for comments to parse into
     * parameter type information.
     */
    static parseParameters(lines: string[]): {params: Parameter[], name: string} {
        let params: Parameter[] = [];
        let name: string|null = null;
        let param: Parameter|null = null;
        for(let line of lines) {
            if (param) {
                if (!param.type) {
                    for (let prefix of [`[A-Z][A-Z0-9_]* is a?`, "\\(workspace\\)"]) {
                        let typeMatch = line.match(`^\\*>\\s+${prefix}([A-Z\\s]+)(\\s*array)?(, dimension\\s*\\((.+)\\))?`);
                        if (typeMatch) {
                            param.type = typeMatch[1].trim();
                            param.isArray = typeMatch[2] != null;
                            param.dimensions = typeMatch[3] ? typeMatch[4].trim() : '';
                            break;
                        }
                    }
                }

                if (line.match(/\\endverbatim/)) {
                    if (!param.type && (param.name.match(/^LT[A-Z]$/) || param.name.match(/^LI?WORK$/))) {
                        param.type = 'INTEGER';
                        param.isArray = false;
                    }

                    params.push(param);
                    param = null;
                }

                continue;
            }

            let paramMatch = line.match(/^\*> \\param\[((?:in)?,?(?:out)?)\] (\w+)/);
            if (paramMatch) {
                param = new Parameter(
                    paramMatch[2],
                    paramMatch[1].indexOf('in') >= 0,
                    paramMatch[1].indexOf('out') >= 0);
                continue;
            }

            let nameMatch = line.match(/^\s*SUBROUTINE (\w+)\s*\(/);
            if (nameMatch) {
                name = nameMatch[1].toLowerCase();
                break;
            }
        }

        // Check the parameters are ok
        for (let param of params) {
            param.validate();
        }

        if (!name) {
            throw new Error("Function name not found.");
        }

        return {params, name};
    }

    /*
     * Prepares the arguments for the TypeScript function that invokes
     * the emLAPACK function.
     */
    static prepareArguments(params: Parameter[]): {
        args: string[],
        returnType: string | null,
        returnParam: Parameter | null,
        hasInfo: boolean,
        hasWork: boolean
    } {
        let args: string[] = [];
        let returnType = null;
        let returnParam = null;
        let hasInfo = false;
        let hasWork = false;
        for (let param of params) {
            if (param.name == 'INFO') {
                hasInfo = true;
                continue;
            }
            else if (param.name.match(/^L[A-Z]*WORK$/)) {
                hasWork = true;
                continue;
            }

            let tsType = constants.FortranTypes.get(param.type);
            if (!tsType) {
                throw new Error(`Missing TypeScript type for FORTRAN type ${param.type}.`);
            }

            if (param.isArray) {
                tsType = `ndarray<${tsType}>`;
            }

            if (param.isIn) {
                args.push(`${param.name}: ${tsType}`);
            }
            else if (param.isOut && !param.name.match(/^[A-Z]*WORK$/) && param.name != 'INFO') {
                returnType = tsType;
                returnParam = param;
            }
        }

        // Check any locals required for the dimensions are present
        let inParams = params.filter(p => p.isIn || p.name.match(/L[A-Z]*WORK/));
        let paramNames = new Set<string>(inParams.map(x => x.name));
        for (let param of params) {
            if (param.hasDimensions()) {
                let {locals} = param.getDimensions();
                for (let l of locals) {
                    if (!paramNames.has(l)) {
                        throw new Error(`Missing parameter '${l}' required for '${param.name}'.`);
                    }
                }
            }
        }

        return {
            args,
            returnType,
            returnParam,
            hasInfo,
            hasWork
        }
    }

    /*
     * Output a generated wrapper file and return the function name
     */
    static processSubroutine(filename: string, outputDirectory: string): string {
        console.log('Reading file ' + filename);
        let text = fs.readFileSync(filename).toString();
        let lines = text.split(/\r?\n/);

        // get the parameters from the code comments
        let {params, name} = Program.parseParameters(lines);

        // figure out which parameters should be uses a TypeScript arguments
        let {args, returnType, returnParam, hasInfo, hasWork} = Program.prepareArguments(params);

        // Write the TypeScript code for calling this function
        let script = `
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function ${name}(${args.join(',\n    ')})${returnType ? ': ' + returnType : ''} {
`;
        script += `
        let func = emlapack.cwrap('${name}_', null, [`;

        for (let p of params) {
            script += `
            'number', // ${p.toString()}`;
        }

        script += `
        ]);
`;

        if (hasInfo) {
            script += `
        let INFO;`;
        }

        // Set any L[X]WORK parameters to -1
        if (hasWork) {
            for (let p of params.filter(p => p.name.match(/^L[A-Z]*WORK$/))) {
                script += `
        let ${p.name} = -1;`;
            }
            script += '\n';
        }

        // allocate the basic parameters
        for (let p of params.filter(p => !p.isArray)) {
                script += `
        let p${p.name} = emlapack._malloc(${constants.SizeOf.get(p.type)});`;
        }

        script += '\n';

        // set the basic input parameter values
        for (let p of params.filter(p => !p.isArray)) {
            let modifier = constants.EmLapackModifiers.get(p.type);
            let value = modifier ? modifier(p.name) : p.name;

            if (p.isIn) {
                script += `
        emlapack.setValue(p${p.name}, ${value}, '${constants.EmLapackTypes.get(p.type)}');`
            }
        }

        script += '\n';

        // allocate and set the array parameters
        for (let p of params.filter(p => p.isArray)) {
            let arrayType = constants.ArrayTypes.get(p.type);
            let heapType = constants.HeapTypes.get(p.type);
            if (!arrayType || !heapType) {
                throw new Error(`Arrays of '${p.type}' are not currently supported`);
            }

            let {dims} = p.getDimensions();
            let dimStr = dims.map(d => '(' + d + ')').join(' * ');

            script += `
        let p${p.name} = emlapack._malloc(${constants.SizeOf.get(p.type)} * ${dimStr});`;

            if (p.isIn) {
                script += `
        let a${p.name} = new ${arrayType}(emlapack.${heapType}.buffer, p${p.name}, ${dimStr});
        a${p.name}.set(${p.name}.data, ${p.name}.offset);`;
            } else {
                script += `
        let ${p.name} = new ${arrayType}(emlapack.${heapType}.buffer, p${p.name}, ${dimStr});`;
            }

            script += '\n';
        }

        // Call the function
        script += `
        func(${params.map(p => 'p' + p.name).join(', ')});`;

        if (hasInfo) {
            let infoParam = params.find(p => p.name == 'INFO');
            if (!infoParam) {
                throw new Error("Missing INFO parameter");
            }
            script += `
        INFO = emlapack.getValue(pINFO, '${constants.EmLapackTypes.get(infoParam.type)}');
        if (INFO != 0) {
            throw new Error("Error calling '${name}': " + INFO);
        }`;
        }

        // If there were work array sizes to be calculated the function will need to be called again
        if (hasWork) {
            script += '\n';
            for (let p of params.filter(p => p.name.match(/^L[A-Z]*WORK$/))) {
                let arrayName = p.name.substr(1);
                let arrayParam = params.find(p => p.name == arrayName);
                if (!arrayParam) {
                    throw new Error("Missing array parameter for " + p.name);
                }

                let {dims} = p.getDimensions();
                let dimStr = dims.map(d => '(' + d + ')').join(' * ');

                script += `
        ${p.name} = emlapack.getValue(p${arrayParam.name}, '${constants.EmLapackTypes.get(arrayParam.type)}');
        emlapack.setValue(p${p.name}, ${p.name}, '${constants.EmLapackTypes.get(p.type)}');
        p${arrayParam.name} = emlapack._malloc(${constants.SizeOf.get(arrayParam.type)} * ${p.name});`;
            }

            script += '\n';

            // Call the function again
            script += `
        func(${params.map(p => 'p' + p.name).join(', ')});`;

            if (hasInfo) {
                let infoParam = params.find(p => p.name == 'INFO');
                if (!infoParam) {
                    throw new Error("Missing INFO parameter");
                }
                script += `
        INFO = emlapack.getValue(pINFO, '${constants.EmLapackTypes.get(infoParam.type)}');
        if (INFO != 0) {
            throw new Error("Error calling '${name}': " + INFO);
        }`;
                script += '\n';
            }
        }

        // Finally return the output
        if (returnParam) {
            if (returnParam.isArray) {
                // convert to a ndarray
                let {dims} = returnParam.getDimensions();
                let dimStr = dims.map(d => '(' + d + ')').join(', ');

                script += `
        return ndarray(${returnParam.name}, [${dimStr}]);`;
            }
            else {
                // fetch the data and return it
                script += `
        return emlapack.getValue(p${returnParam.name}, '${constants.EmLapackTypes.get(returnParam.type)}');`;
            }
        }

        // End the function
        script += `
}`;

        // Write the output TypeScript file
        if (!fs.existsSync(outputDirectory)) {
           fs.mkdirSync(outputDirectory, {recursive: true});
        }

        let outfile = path.join(outputDirectory, name + '.ts');
        fs.writeFileSync(outfile, script);

        return name;
    }

    static writeExportsFile(functions: string[], outputDirectory: string, filename: string = 'tslapack.ts') {
        let script = '';
        for (let f of functions) {
            script += `import * as _${f} from "./functions/${f}";\n`;
        }

        for (let f of functions) {
            script += `export import ${f} = _${f}.${f};\n`;
        }

        if (!fs.existsSync(outputDirectory)) {
            fs.mkdirSync(outputDirectory, {recursive: true});
        }

        let outfile = path.join(outputDirectory, filename);
        fs.writeFileSync(outfile, script);
    }
}

const p = new Program();
