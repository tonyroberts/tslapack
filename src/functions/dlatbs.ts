
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlatbs(UPLO: string,
    TRANS: string,
    DIAG: string,
    NORMIN: string,
    N: number,
    KD: number,
    AB: ndarray<number>,
    LDAB: number,
    X: ndarray<number>,
    CNORM: ndarray<number>): number {

        let func = emlapack.cwrap('dlatbs_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] TRANS: CHARACTER
            'number', // [in] DIAG: CHARACTER
            'number', // [in] NORMIN: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
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
        let pKD = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pSCALE = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pDIAG, DIAG.charCodeAt(0), 'i8');
        emlapack.setValue(pNORMIN, NORMIN.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pX = emlapack._malloc(8 * (N));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (N));
        aX.set(X.data, X.offset);

        let pCNORM = emlapack._malloc(8 * (N));
        let aCNORM = new Float64Array(emlapack.HEAPF64.buffer, pCNORM, (N));
        aCNORM.set(CNORM.data, CNORM.offset);

        func(pUPLO, pTRANS, pDIAG, pNORMIN, pN, pKD, pAB, pLDAB, pX, pSCALE, pCNORM, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlatbs': " + INFO);
        }
        return emlapack.getValue(pSCALE, 'double');
}