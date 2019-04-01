
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dptsvx(FACT: string,
    N: number,
    NRHS: number,
    D: ndarray<number>,
    E: ndarray<number>,
    DF: ndarray<number>,
    EF: ndarray<number>,
    B: ndarray<number>,
    LDB: number,
    LDX: number): ndarray<number> {

        let func = emlapack.cwrap('dptsvx_', null, [
            'number', // [in] FACT: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E: DOUBLE PRECISION[N-1]
            'number', // [in,out] DF: DOUBLE PRECISION[N]
            'number', // [in,out] EF: DOUBLE PRECISION[N-1]
            'number', // [in] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [out] X: DOUBLE PRECISION[LDX,NRHS]
            'number', // [in] LDX: INTEGER
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] FERR: DOUBLE PRECISION[NRHS]
            'number', // [out] BERR: DOUBLE PRECISION[NRHS]
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pFACT = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDX = emlapack._malloc(4);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pFACT, FACT.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDX, LDX, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pDF = emlapack._malloc(8 * (N));
        let aDF = new Float64Array(emlapack.HEAPF64.buffer, pDF, (N));
        aDF.set(DF.data, DF.offset);

        let pEF = emlapack._malloc(8 * (N-1));
        let aEF = new Float64Array(emlapack.HEAPF64.buffer, pEF, (N-1));
        aEF.set(EF.data, EF.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        let pX = emlapack._malloc(8 * (LDX) * (NRHS));
        let X = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (NRHS));

        let pFERR = emlapack._malloc(8 * (NRHS));
        let FERR = new Float64Array(emlapack.HEAPF64.buffer, pFERR, (NRHS));

        let pBERR = emlapack._malloc(8 * (NRHS));
        let BERR = new Float64Array(emlapack.HEAPF64.buffer, pBERR, (NRHS));

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        func(pFACT, pN, pNRHS, pD, pE, pDF, pEF, pB, pLDB, pX, pLDX, pRCOND, pFERR, pBERR, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dptsvx': " + INFO);
        }
        return ndarray(BERR, [(NRHS)]);
}