import * as constants from "./constants";

/**
 * Parameter information parsed from LAPACK Fortran comments.
 */
export class Parameter {
    name: string;
    type: string;
    isIn: boolean;
    isOut: boolean;
    isArray: boolean;
    dimensions: string;

    constructor(name: string, isIn: boolean, isOut: boolean, type: string='', isArray: boolean=false, dimensions: string='') {
        this.name = name;
        this.isIn = isIn;
        this.isOut = isOut;
        this.type = type;
        this.isArray = isArray;
        this.dimensions = dimensions;
    }

    toString(): string {
        let x = '';
        if (this.isIn && this.isOut) {
            x += '[in,out] ';
        } else if (this.isIn) {
            x += '[in] ';
        }
        else if (this.isOut) {
            x += '[out] ';
        }
        x += this.name;
        if (this.type) {
            x += ': ' + this.type;
            if (this.isArray) {
                if (this.dimensions) {
                    x += "[" + this.dimensions + "]"
                }
                else {
                    x += '(unknown)';
                }
            }
        }
        return x;
    }

    validate(): void {
        if (!this.type) {
            throw new Error(`Parameter '${this.name}' has no type.`);
        }

        if (constants.UnsupportedTypes.get(this.type)) {
            throw new Error(`Parameter '${this.name}' has an unsupported type.`);
        }

        if (!constants.FortranTypes.get(this.type)) {
            throw new Error(`Parameter '${this.name}' has unknown type '${this.type}'.`);
        }

        if (this.isArray && !this.dimensions) {
            throw new Error(`Parameter '${this.name}' is an array with unknown dimensions.`);
        }
    }

    hasDimensions(): boolean {
        return this.dimensions != '';
    }

    getDimensions(): {dims: string[], locals: Set<string>} {
        let locals = new Set<string>();

        if (!this.hasDimensions()) {
            return {dims: [], locals: locals};
        }

        let inParens = 0;
        let curr = '';
        let dims = [];
        for (let c of this.dimensions) {
            if (c == ' ' || c == '\t') {
                continue;
            }
            if (c == '(') {
                inParens++;
            }
            else if (c == ')') {
                inParens--;
            }

            if (c == ',' && !inParens) {
                dims.push(curr);
                curr = '';
            }
            else {
                curr += c;
            }
        }

        if (curr) {
            // Some arguments have a trailing ')' (eg DBDSVDX).
            if (inParens < 0) {
                curr = curr.substr(0, curr.length + inParens);
            }
            dims.push(curr);
        }

        dims.forEach(d => {
            d.replace(/[A-Z_][A-Z0-9]*/gi, s => {
                let l = s.toLowerCase();
                if (l != 'min' && l != 'max') {
                    locals.add(s);
                }
                return l;
            })
        })

        dims = dims.map(d => d.replace(/min\(/gi, x => 'Math.min('))
                   .map(d => d.replace(/max\(/gi, x => 'Math.max('));

        return {dims, locals}
    }
}
