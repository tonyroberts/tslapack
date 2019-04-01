
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlatrs(UPLO: string,
    TRANS: string,
    DIAG: string,
    NORMIN: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    X: ndarray<number>,
    CNORM: ndarray<number>): number {

        let func = emlapack.cwrap('dlatrs_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] TRANS: CHARACTER
            'number', // [in] DIAG: CHARACTER
            'number', // [in] NORMIN: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] X: DOUBLE PRECISION[N]
            'number', // [out] SCALE: DOUBLE PRECISION
            'number', // [in,out] CNORM: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pTRANS = emlapack._malloc(1);
        let pDIAG = emlapack._malloc(1);
        let pNORMIN = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pSCALE = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pDIAG, DIAG.charCodeAt(0), 'i8');
        emlapack.setValue(pNORMIN, NORMIN.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pX = emlapack._malloc(8 * (N));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (N));
        aX.set(X.data, X.offset);

        let pCNORM = emlapack._malloc(8 * (N));
        let aCNORM = new Float64Array(emlapack.HEAPF64.buffer, pCNORM, (N));
        aCNORM.set(CNORM.data, CNORM.offset);

        func(pUPLO, pTRANS, pDIAG, pNORMIN, pN, pA, pLDA, pX, pSCALE, pCNORM, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlatrs': " + INFO);
        }
        return emlapack.getValue(pSCALE, 'double');
}