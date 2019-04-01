
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarre(RANGE: string,
    N: number,
    VL: number,
    VU: number,
    IL: number,
    IU: number,
    D: ndarray<number>,
    E: ndarray<number>,
    E2: ndarray<number>,
    RTOL1: number,
    RTOL2: number,
    SPLTOL: number): number {

        let func = emlapack.cwrap('dlarre_', null, [
            'number', // [in] RANGE: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] VL: DOUBLE PRECISION
            'number', // [in,out] VU: DOUBLE PRECISION
            'number', // [in] IL: INTEGER
            'number', // [in] IU: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N]
            'number', // [in,out] E2: DOUBLE PRECISION[N]
            'number', // [in] RTOL1: DOUBLE PRECISION
            'number', // [in] RTOL2: DOUBLE PRECISION
            'number', // [in] SPLTOL: DOUBLE PRECISION
            'number', // [out] NSPLIT: INTEGER
            'number', // [out] ISPLIT: INTEGER[N]
            'number', // [out] M: INTEGER
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] WERR: DOUBLE PRECISION[N]
            'number', // [out] WGAP: DOUBLE PRECISION[N]
            'number', // [out] IBLOCK: INTEGER[N]
            'number', // [out] INDEXW: INTEGER[N]
            'number', // [out] GERS: DOUBLE PRECISION[2*N]
            'number', // [out] PIVMIN: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[6*N]
            'number', // [out] IWORK: INTEGER[5*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pRANGE = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pVL = emlapack._malloc(8);
        let pVU = emlapack._malloc(8);
        let pIL = emlapack._malloc(4);
        let pIU = emlapack._malloc(4);
        let pRTOL1 = emlapack._malloc(8);
        let pRTOL2 = emlapack._malloc(8);
        let pSPLTOL = emlapack._malloc(8);
        let pNSPLIT = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pPIVMIN = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pRANGE, RANGE.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pVL, VL, 'double');
        emlapack.setValue(pVU, VU, 'double');
        emlapack.setValue(pIL, IL, 'i32');
        emlapack.setValue(pIU, IU, 'i32');
        emlapack.setValue(pRTOL1, RTOL1, 'double');
        emlapack.setValue(pRTOL2, RTOL2, 'double');
        emlapack.setValue(pSPLTOL, SPLTOL, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N));
        aE.set(E.data, E.offset);

        let pE2 = emlapack._malloc(8 * (N));
        let aE2 = new Float64Array(emlapack.HEAPF64.buffer, pE2, (N));
        aE2.set(E2.data, E2.offset);

        let pISPLIT = emlapack._malloc(4 * (N));
        let ISPLIT = new Int32Array(emlapack.HEAPI32.buffer, pISPLIT, (N));

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pWERR = emlapack._malloc(8 * (N));
        let WERR = new Float64Array(emlapack.HEAPF64.buffer, pWERR, (N));

        let pWGAP = emlapack._malloc(8 * (N));
        let WGAP = new Float64Array(emlapack.HEAPF64.buffer, pWGAP, (N));

        let pIBLOCK = emlapack._malloc(4 * (N));
        let IBLOCK = new Int32Array(emlapack.HEAPI32.buffer, pIBLOCK, (N));

        let pINDEXW = emlapack._malloc(4 * (N));
        let INDEXW = new Int32Array(emlapack.HEAPI32.buffer, pINDEXW, (N));

        let pGERS = emlapack._malloc(8 * (2*N));
        let GERS = new Float64Array(emlapack.HEAPF64.buffer, pGERS, (2*N));

        let pWORK = emlapack._malloc(8 * (6*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (6*N));

        let pIWORK = emlapack._malloc(4 * (5*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (5*N));

        func(pRANGE, pN, pVL, pVU, pIL, pIU, pD, pE, pE2, pRTOL1, pRTOL2, pSPLTOL, pNSPLIT, pISPLIT, pM, pW, pWERR, pWGAP, pIBLOCK, pINDEXW, pGERS, pPIVMIN, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarre': " + INFO);
        }
        return emlapack.getValue(pPIVMIN, 'double');
}