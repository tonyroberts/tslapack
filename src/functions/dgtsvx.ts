
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgtsvx(FACT: string,
    TRANS: string,
    N: number,
    NRHS: number,
    DL: ndarray<number>,
    D: ndarray<number>,
    DU: ndarray<number>,
    DLF: ndarray<number>,
    DF: ndarray<number>,
    DUF: ndarray<number>,
    DU2: ndarray<number>,
    IPIV: ndarray<number>,
    B: ndarray<number>,
    LDB: number,
    LDX: number): ndarray<number> {

        let func = emlapack.cwrap('dgtsvx_', null, [
            'number', // [in] FACT: CHARACTER
            'number', // [in] TRANS: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] DL: DOUBLE PRECISION[N-1]
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] DU: DOUBLE PRECISION[N-1]
            'number', // [in,out] DLF: DOUBLE PRECISION[N-1]
            'number', // [in,out] DF: DOUBLE PRECISION[N]
            'number', // [in,out] DUF: DOUBLE PRECISION[N-1]
            'number', // [in,out] DU2: DOUBLE PRECISION[N-2]
            'number', // [in,out] IPIV: INTEGER[N]
            'number', // [in] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [out] X: DOUBLE PRECISION[LDX,NRHS]
            'number', // [in] LDX: INTEGER
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] FERR: DOUBLE PRECISION[NRHS]
            'number', // [out] BERR: DOUBLE PRECISION[NRHS]
            'number', // [out] WORK: DOUBLE PRECISION[3*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pFACT = emlapack._malloc(1);
        let pTRANS = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDX = emlapack._malloc(4);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pFACT, FACT.charCodeAt(0), 'i8');
        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDX, LDX, 'i32');

        let pDL = emlapack._malloc(8 * (N-1));
        let aDL = new Float64Array(emlapack.HEAPF64.buffer, pDL, (N-1));
        aDL.set(DL.data, DL.offset);

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pDU = emlapack._malloc(8 * (N-1));
        let aDU = new Float64Array(emlapack.HEAPF64.buffer, pDU, (N-1));
        aDU.set(DU.data, DU.offset);

        let pDLF = emlapack._malloc(8 * (N-1));
        let aDLF = new Float64Array(emlapack.HEAPF64.buffer, pDLF, (N-1));
        aDLF.set(DLF.data, DLF.offset);

        let pDF = emlapack._malloc(8 * (N));
        let aDF = new Float64Array(emlapack.HEAPF64.buffer, pDF, (N));
        aDF.set(DF.data, DF.offset);

        let pDUF = emlapack._malloc(8 * (N-1));
        let aDUF = new Float64Array(emlapack.HEAPF64.buffer, pDUF, (N-1));
        aDUF.set(DUF.data, DUF.offset);

        let pDU2 = emlapack._malloc(8 * (N-2));
        let aDU2 = new Float64Array(emlapack.HEAPF64.buffer, pDU2, (N-2));
        aDU2.set(DU2.data, DU2.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        let pX = emlapack._malloc(8 * (LDX) * (NRHS));
        let X = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (NRHS));

        let pFERR = emlapack._malloc(8 * (NRHS));
        let FERR = new Float64Array(emlapack.HEAPF64.buffer, pFERR, (NRHS));

        let pBERR = emlapack._malloc(8 * (NRHS));
        let BERR = new Float64Array(emlapack.HEAPF64.buffer, pBERR, (NRHS));

        let pWORK = emlapack._malloc(8 * (3*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (3*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pFACT, pTRANS, pN, pNRHS, pDL, pD, pDU, pDLF, pDF, pDUF, pDU2, pIPIV, pB, pLDB, pX, pLDX, pRCOND, pFERR, pBERR, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgtsvx': " + INFO);
        }
        return ndarray(BERR, [(NRHS)]);
}