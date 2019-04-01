
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarrj(N: number,
    D: ndarray<number>,
    E2: ndarray<number>,
    IFIRST: number,
    ILAST: number,
    RTOL: number,
    OFFSET: number,
    W: ndarray<number>,
    WERR: ndarray<number>,
    PIVMIN: number,
    SPDIAM: number) {

        let func = emlapack.cwrap('dlarrj_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E2: DOUBLE PRECISION[N-1]
            'number', // [in] IFIRST: INTEGER
            'number', // [in] ILAST: INTEGER
            'number', // [in] RTOL: DOUBLE PRECISION
            'number', // [in] OFFSET: INTEGER
            'number', // [in,out] W: DOUBLE PRECISION[N]
            'number', // [in,out] WERR: DOUBLE PRECISION[N]
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] IWORK: INTEGER[2*N]
            'number', // [in] PIVMIN: DOUBLE PRECISION
            'number', // [in] SPDIAM: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pIFIRST = emlapack._malloc(4);
        let pILAST = emlapack._malloc(4);
        let pRTOL = emlapack._malloc(8);
        let pOFFSET = emlapack._malloc(4);
        let pPIVMIN = emlapack._malloc(8);
        let pSPDIAM = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pIFIRST, IFIRST, 'i32');
        emlapack.setValue(pILAST, ILAST, 'i32');
        emlapack.setValue(pRTOL, RTOL, 'double');
        emlapack.setValue(pOFFSET, OFFSET, 'i32');
        emlapack.setValue(pPIVMIN, PIVMIN, 'double');
        emlapack.setValue(pSPDIAM, SPDIAM, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE2 = emlapack._malloc(8 * (N-1));
        let aE2 = new Float64Array(emlapack.HEAPF64.buffer, pE2, (N-1));
        aE2.set(E2.data, E2.offset);

        let pW = emlapack._malloc(8 * (N));
        let aW = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));
        aW.set(W.data, W.offset);

        let pWERR = emlapack._malloc(8 * (N));
        let aWERR = new Float64Array(emlapack.HEAPF64.buffer, pWERR, (N));
        aWERR.set(WERR.data, WERR.offset);

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        let pIWORK = emlapack._malloc(4 * (2*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (2*N));

        func(pN, pD, pE2, pIFIRST, pILAST, pRTOL, pOFFSET, pW, pWERR, pWORK, pIWORK, pPIVMIN, pSPDIAM, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarrj': " + INFO);
        }
}