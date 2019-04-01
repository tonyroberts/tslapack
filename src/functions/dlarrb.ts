
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarrb(N: number,
    D: ndarray<number>,
    LLD: ndarray<number>,
    IFIRST: number,
    ILAST: number,
    RTOL1: number,
    RTOL2: number,
    OFFSET: number,
    W: ndarray<number>,
    WGAP: ndarray<number>,
    WERR: ndarray<number>,
    PIVMIN: number,
    SPDIAM: number,
    TWIST: number) {

        let func = emlapack.cwrap('dlarrb_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] LLD: DOUBLE PRECISION[N-1]
            'number', // [in] IFIRST: INTEGER
            'number', // [in] ILAST: INTEGER
            'number', // [in] RTOL1: DOUBLE PRECISION
            'number', // [in] RTOL2: DOUBLE PRECISION
            'number', // [in] OFFSET: INTEGER
            'number', // [in,out] W: DOUBLE PRECISION[N]
            'number', // [in,out] WGAP: DOUBLE PRECISION[N-1]
            'number', // [in,out] WERR: DOUBLE PRECISION[N]
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] IWORK: INTEGER[2*N]
            'number', // [in] PIVMIN: DOUBLE PRECISION
            'number', // [in] SPDIAM: DOUBLE PRECISION
            'number', // [in] TWIST: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pIFIRST = emlapack._malloc(4);
        let pILAST = emlapack._malloc(4);
        let pRTOL1 = emlapack._malloc(8);
        let pRTOL2 = emlapack._malloc(8);
        let pOFFSET = emlapack._malloc(4);
        let pPIVMIN = emlapack._malloc(8);
        let pSPDIAM = emlapack._malloc(8);
        let pTWIST = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pIFIRST, IFIRST, 'i32');
        emlapack.setValue(pILAST, ILAST, 'i32');
        emlapack.setValue(pRTOL1, RTOL1, 'double');
        emlapack.setValue(pRTOL2, RTOL2, 'double');
        emlapack.setValue(pOFFSET, OFFSET, 'i32');
        emlapack.setValue(pPIVMIN, PIVMIN, 'double');
        emlapack.setValue(pSPDIAM, SPDIAM, 'double');
        emlapack.setValue(pTWIST, TWIST, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pLLD = emlapack._malloc(8 * (N-1));
        let aLLD = new Float64Array(emlapack.HEAPF64.buffer, pLLD, (N-1));
        aLLD.set(LLD.data, LLD.offset);

        let pW = emlapack._malloc(8 * (N));
        let aW = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));
        aW.set(W.data, W.offset);

        let pWGAP = emlapack._malloc(8 * (N-1));
        let aWGAP = new Float64Array(emlapack.HEAPF64.buffer, pWGAP, (N-1));
        aWGAP.set(WGAP.data, WGAP.offset);

        let pWERR = emlapack._malloc(8 * (N));
        let aWERR = new Float64Array(emlapack.HEAPF64.buffer, pWERR, (N));
        aWERR.set(WERR.data, WERR.offset);

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        let pIWORK = emlapack._malloc(4 * (2*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (2*N));

        func(pN, pD, pLLD, pIFIRST, pILAST, pRTOL1, pRTOL2, pOFFSET, pW, pWGAP, pWERR, pWORK, pIWORK, pPIVMIN, pSPDIAM, pTWIST, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarrb': " + INFO);
        }
}