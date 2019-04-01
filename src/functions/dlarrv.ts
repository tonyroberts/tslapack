
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarrv(N: number,
    VL: number,
    VU: number,
    D: ndarray<number>,
    L: ndarray<number>,
    PIVMIN: number,
    ISPLIT: ndarray<number>,
    M: number,
    DOL: number,
    DOU: number,
    MINRGP: number,
    RTOL1: number,
    RTOL2: number,
    W: ndarray<number>,
    WERR: ndarray<number>,
    WGAP: ndarray<number>,
    IBLOCK: ndarray<number>,
    INDEXW: ndarray<number>,
    GERS: ndarray<number>,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dlarrv_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] VL: DOUBLE PRECISION
            'number', // [in] VU: DOUBLE PRECISION
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] L: DOUBLE PRECISION[N]
            'number', // [in] PIVMIN: DOUBLE PRECISION
            'number', // [in] ISPLIT: INTEGER[N]
            'number', // [in] M: INTEGER
            'number', // [in] DOL: INTEGER
            'number', // [in] DOU: INTEGER
            'number', // [in] MINRGP: DOUBLE PRECISION
            'number', // [in] RTOL1: DOUBLE PRECISION
            'number', // [in] RTOL2: DOUBLE PRECISION
            'number', // [in,out] W: DOUBLE PRECISION[N]
            'number', // [in,out] WERR: DOUBLE PRECISION[N]
            'number', // [in,out] WGAP: DOUBLE PRECISION[N]
            'number', // [in] IBLOCK: INTEGER[N]
            'number', // [in] INDEXW: INTEGER[N]
            'number', // [in] GERS: DOUBLE PRECISION[2*N]
            'number', // [out] Z: DOUBLE PRECISION[LDZ, max(1,M)]
            'number', // [in] LDZ: INTEGER
            'number', // [out] ISUPPZ: INTEGER[2*max(1,M)]
            'number', // [out] WORK: DOUBLE PRECISION[12*N]
            'number', // [out] IWORK: INTEGER[7*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pVL = emlapack._malloc(8);
        let pVU = emlapack._malloc(8);
        let pPIVMIN = emlapack._malloc(8);
        let pM = emlapack._malloc(4);
        let pDOL = emlapack._malloc(4);
        let pDOU = emlapack._malloc(4);
        let pMINRGP = emlapack._malloc(8);
        let pRTOL1 = emlapack._malloc(8);
        let pRTOL2 = emlapack._malloc(8);
        let pLDZ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pVL, VL, 'double');
        emlapack.setValue(pVU, VU, 'double');
        emlapack.setValue(pPIVMIN, PIVMIN, 'double');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pDOL, DOL, 'i32');
        emlapack.setValue(pDOU, DOU, 'i32');
        emlapack.setValue(pMINRGP, MINRGP, 'double');
        emlapack.setValue(pRTOL1, RTOL1, 'double');
        emlapack.setValue(pRTOL2, RTOL2, 'double');
        emlapack.setValue(pLDZ, LDZ, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pL = emlapack._malloc(8 * (N));
        let aL = new Float64Array(emlapack.HEAPF64.buffer, pL, (N));
        aL.set(L.data, L.offset);

        let pISPLIT = emlapack._malloc(4 * (N));
        let aISPLIT = new Int32Array(emlapack.HEAPI32.buffer, pISPLIT, (N));
        aISPLIT.set(ISPLIT.data, ISPLIT.offset);

        let pW = emlapack._malloc(8 * (N));
        let aW = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));
        aW.set(W.data, W.offset);

        let pWERR = emlapack._malloc(8 * (N));
        let aWERR = new Float64Array(emlapack.HEAPF64.buffer, pWERR, (N));
        aWERR.set(WERR.data, WERR.offset);

        let pWGAP = emlapack._malloc(8 * (N));
        let aWGAP = new Float64Array(emlapack.HEAPF64.buffer, pWGAP, (N));
        aWGAP.set(WGAP.data, WGAP.offset);

        let pIBLOCK = emlapack._malloc(4 * (N));
        let aIBLOCK = new Int32Array(emlapack.HEAPI32.buffer, pIBLOCK, (N));
        aIBLOCK.set(IBLOCK.data, IBLOCK.offset);

        let pINDEXW = emlapack._malloc(4 * (N));
        let aINDEXW = new Int32Array(emlapack.HEAPI32.buffer, pINDEXW, (N));
        aINDEXW.set(INDEXW.data, INDEXW.offset);

        let pGERS = emlapack._malloc(8 * (2*N));
        let aGERS = new Float64Array(emlapack.HEAPF64.buffer, pGERS, (2*N));
        aGERS.set(GERS.data, GERS.offset);

        let pZ = emlapack._malloc(8 * (LDZ) * (Math.max(1,M)));
        let Z = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (Math.max(1,M)));

        let pISUPPZ = emlapack._malloc(4 * (2*Math.max(1,M)));
        let ISUPPZ = new Int32Array(emlapack.HEAPI32.buffer, pISUPPZ, (2*Math.max(1,M)));

        let pWORK = emlapack._malloc(8 * (12*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (12*N));

        let pIWORK = emlapack._malloc(4 * (7*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (7*N));

        func(pN, pVL, pVU, pD, pL, pPIVMIN, pISPLIT, pM, pDOL, pDOU, pMINRGP, pRTOL1, pRTOL2, pW, pWERR, pWGAP, pIBLOCK, pINDEXW, pGERS, pZ, pLDZ, pISUPPZ, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarrv': " + INFO);
        }
        return ndarray(ISUPPZ, [(2*Math.max(1,M))]);
}