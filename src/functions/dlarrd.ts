
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarrd(RANGE: string,
    ORDER: string,
    N: number,
    VL: number,
    VU: number,
    IL: number,
    IU: number,
    GERS: ndarray<number>,
    RELTOL: number,
    D: ndarray<number>,
    E: ndarray<number>,
    E2: ndarray<number>,
    PIVMIN: number,
    NSPLIT: number,
    ISPLIT: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dlarrd_', null, [
            'number', // [in] RANGE: CHARACTER
            'number', // [in] ORDER: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] VL: DOUBLE PRECISION
            'number', // [in] VU: DOUBLE PRECISION
            'number', // [in] IL: INTEGER
            'number', // [in] IU: INTEGER
            'number', // [in] GERS: DOUBLE PRECISION[2*N]
            'number', // [in] RELTOL: DOUBLE PRECISION
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E: DOUBLE PRECISION[N-1]
            'number', // [in] E2: DOUBLE PRECISION[N-1]
            'number', // [in] PIVMIN: DOUBLE PRECISION
            'number', // [in] NSPLIT: INTEGER
            'number', // [in] ISPLIT: INTEGER[N]
            'number', // [out] M: INTEGER
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] WERR: DOUBLE PRECISION[N]
            'number', // [out] WL: DOUBLE PRECISION
            'number', // [out] WU: DOUBLE PRECISION
            'number', // [out] IBLOCK: INTEGER[N]
            'number', // [out] INDEXW: INTEGER[N]
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
        let pRELTOL = emlapack._malloc(8);
        let pPIVMIN = emlapack._malloc(8);
        let pNSPLIT = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pWL = emlapack._malloc(8);
        let pWU = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pRANGE, RANGE.charCodeAt(0), 'i8');
        emlapack.setValue(pORDER, ORDER.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pVL, VL, 'double');
        emlapack.setValue(pVU, VU, 'double');
        emlapack.setValue(pIL, IL, 'i32');
        emlapack.setValue(pIU, IU, 'i32');
        emlapack.setValue(pRELTOL, RELTOL, 'double');
        emlapack.setValue(pPIVMIN, PIVMIN, 'double');
        emlapack.setValue(pNSPLIT, NSPLIT, 'i32');

        let pGERS = emlapack._malloc(8 * (2*N));
        let aGERS = new Float64Array(emlapack.HEAPF64.buffer, pGERS, (2*N));
        aGERS.set(GERS.data, GERS.offset);

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pE2 = emlapack._malloc(8 * (N-1));
        let aE2 = new Float64Array(emlapack.HEAPF64.buffer, pE2, (N-1));
        aE2.set(E2.data, E2.offset);

        let pISPLIT = emlapack._malloc(4 * (N));
        let aISPLIT = new Int32Array(emlapack.HEAPI32.buffer, pISPLIT, (N));
        aISPLIT.set(ISPLIT.data, ISPLIT.offset);

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pWERR = emlapack._malloc(8 * (N));
        let WERR = new Float64Array(emlapack.HEAPF64.buffer, pWERR, (N));

        let pIBLOCK = emlapack._malloc(4 * (N));
        let IBLOCK = new Int32Array(emlapack.HEAPI32.buffer, pIBLOCK, (N));

        let pINDEXW = emlapack._malloc(4 * (N));
        let INDEXW = new Int32Array(emlapack.HEAPI32.buffer, pINDEXW, (N));

        let pWORK = emlapack._malloc(8 * (4*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (4*N));

        let pIWORK = emlapack._malloc(4 * (3*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (3*N));

        func(pRANGE, pORDER, pN, pVL, pVU, pIL, pIU, pGERS, pRELTOL, pD, pE, pE2, pPIVMIN, pNSPLIT, pISPLIT, pM, pW, pWERR, pWL, pWU, pIBLOCK, pINDEXW, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarrd': " + INFO);
        }
        return ndarray(INDEXW, [(N)]);
}