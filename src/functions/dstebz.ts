
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dstebz(RANGE: string,
    ORDER: string,
    N: number,
    VL: number,
    VU: number,
    IL: number,
    IU: number,
    ABSTOL: number,
    D: ndarray<number>,
    E: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dstebz_', null, [
            'number', // [in] RANGE: CHARACTER
            'number', // [in] ORDER: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] VL: DOUBLE PRECISION
            'number', // [in] VU: DOUBLE PRECISION
            'number', // [in] IL: INTEGER
            'number', // [in] IU: INTEGER
            'number', // [in] ABSTOL: DOUBLE PRECISION
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E: DOUBLE PRECISION[N-1]
            'number', // [out] M: INTEGER
            'number', // [out] NSPLIT: INTEGER
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] IBLOCK: INTEGER[N]
            'number', // [out] ISPLIT: INTEGER[N]
            'number', // [out] WORK: DOUBLE PRECISION[4*N]
            'number', // [out] IWORK: INTEGER[3*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pRANGE = emlapack._malloc(1);
        let pORDER = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pVL = emlapack._malloc(8);
        let pVU = emlapack._malloc(8);
        let pIL = emlapack._malloc(4);
        let pIU = emlapack._malloc(4);
        let pABSTOL = emlapack._malloc(8);
        let pM = emlapack._malloc(4);
        let pNSPLIT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pRANGE, RANGE.charCodeAt(0), 'i8');
        emlapack.setValue(pORDER, ORDER.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pVL, VL, 'double');
        emlapack.setValue(pVU, VU, 'double');
        emlapack.setValue(pIL, IL, 'i32');
        emlapack.setValue(pIU, IU, 'i32');
        emlapack.setValue(pABSTOL, ABSTOL, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pIBLOCK = emlapack._malloc(4 * (N));
        let IBLOCK = new Int32Array(emlapack.HEAPI32.buffer, pIBLOCK, (N));

        let pISPLIT = emlapack._malloc(4 * (N));
        let ISPLIT = new Int32Array(emlapack.HEAPI32.buffer, pISPLIT, (N));

        let pWORK = emlapack._malloc(8 * (4*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (4*N));

        let pIWORK = emlapack._malloc(4 * (3*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (3*N));

        func(pRANGE, pORDER, pN, pVL, pVU, pIL, pIU, pABSTOL, pD, pE, pM, pNSPLIT, pW, pIBLOCK, pISPLIT, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dstebz': " + INFO);
        }
        return ndarray(ISPLIT, [(N)]);
}