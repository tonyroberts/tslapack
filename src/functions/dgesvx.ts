
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgesvx(FACT: string,
    TRANS: string,
    N: number,
    NRHS: number,
    A: ndarray<number>,
    LDA: number,
    AF: ndarray<number>,
    LDAF: number,
    IPIV: ndarray<number>,
    EQUED: string,
    R: ndarray<number>,
    C: ndarray<number>,
    B: ndarray<number>,
    LDB: number,
    LDX: number): ndarray<number> {

        let func = emlapack.cwrap('dgesvx_', null, [
            'number', // [in] FACT: CHARACTER
            'number', // [in] TRANS: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] AF: DOUBLE PRECISION[LDAF,N]
            'number', // [in] LDAF: INTEGER
            'number', // [in,out] IPIV: INTEGER[N]
            'number', // [in,out] EQUED: CHARACTER
            'number', // [in,out] R: DOUBLE PRECISION[N]
            'number', // [in,out] C: DOUBLE PRECISION[N]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [out] X: DOUBLE PRECISION[LDX,NRHS]
            'number', // [in] LDX: INTEGER
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] FERR: DOUBLE PRECISION[NRHS]
            'number', // [out] BERR: DOUBLE PRECISION[NRHS]
            'number', // [out] WORK: DOUBLE PRECISION[4*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pFACT = emlapack._malloc(1);
        let pTRANS = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDAF = emlapack._malloc(4);
        let pEQUED = emlapack._malloc(1);
        let pLDB = emlapack._malloc(4);
        let pLDX = emlapack._malloc(4);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pFACT, FACT.charCodeAt(0), 'i8');
        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDAF, LDAF, 'i32');
        emlapack.setValue(pEQUED, EQUED.charCodeAt(0), 'i8');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDX, LDX, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pAF = emlapack._malloc(8 * (LDAF) * (N));
        let aAF = new Float64Array(emlapack.HEAPF64.buffer, pAF, (LDAF) * (N));
        aAF.set(AF.data, AF.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pR = emlapack._malloc(8 * (N));
        let aR = new Float64Array(emlapack.HEAPF64.buffer, pR, (N));
        aR.set(R.data, R.offset);

        let pC = emlapack._malloc(8 * (N));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (N));
        aC.set(C.data, C.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        let pX = emlapack._malloc(8 * (LDX) * (NRHS));
        let X = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (NRHS));

        let pFERR = emlapack._malloc(8 * (NRHS));
        let FERR = new Float64Array(emlapack.HEAPF64.buffer, pFERR, (NRHS));

        let pBERR = emlapack._malloc(8 * (NRHS));
        let BERR = new Float64Array(emlapack.HEAPF64.buffer, pBERR, (NRHS));

        let pWORK = emlapack._malloc(8 * (4*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (4*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pFACT, pTRANS, pN, pNRHS, pA, pLDA, pAF, pLDAF, pIPIV, pEQUED, pR, pC, pB, pLDB, pX, pLDX, pRCOND, pFERR, pBERR, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgesvx': " + INFO);
        }
        return ndarray(BERR, [(NRHS)]);
}