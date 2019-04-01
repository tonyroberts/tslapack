
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarrk(N: number,
    IW: number,
    GL: number,
    GU: number,
    D: ndarray<number>,
    E2: ndarray<number>,
    PIVMIN: number,
    RELTOL: number): number {

        let func = emlapack.cwrap('dlarrk_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] IW: INTEGER
            'number', // [in] GL: DOUBLE PRECISION
            'number', // [in] GU: DOUBLE PRECISION
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E2: DOUBLE PRECISION[N-1]
            'number', // [in] PIVMIN: DOUBLE PRECISION
            'number', // [in] RELTOL: DOUBLE PRECISION
            'number', // [out] W: DOUBLE PRECISION
            'number', // [out] WERR: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pIW = emlapack._malloc(4);
        let pGL = emlapack._malloc(8);
        let pGU = emlapack._malloc(8);
        let pPIVMIN = emlapack._malloc(8);
        let pRELTOL = emlapack._malloc(8);
        let pW = emlapack._malloc(8);
        let pWERR = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pIW, IW, 'i32');
        emlapack.setValue(pGL, GL, 'double');
        emlapack.setValue(pGU, GU, 'double');
        emlapack.setValue(pPIVMIN, PIVMIN, 'double');
        emlapack.setValue(pRELTOL, RELTOL, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE2 = emlapack._malloc(8 * (N-1));
        let aE2 = new Float64Array(emlapack.HEAPF64.buffer, pE2, (N-1));
        aE2.set(E2.data, E2.offset);

        func(pN, pIW, pGL, pGU, pD, pE2, pPIVMIN, pRELTOL, pW, pWERR, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarrk': " + INFO);
        }
        return emlapack.getValue(pWERR, 'double');
}