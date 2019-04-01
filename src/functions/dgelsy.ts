
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgelsy(M: number,
    N: number,
    NRHS: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    JPVT: ndarray<number>,
    RCOND: number): number {

        let func = emlapack.cwrap('dgelsy_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [in,out] JPVT: INTEGER[N]
            'number', // [in] RCOND: DOUBLE PRECISION
            'number', // [out] RANK: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pRCOND = emlapack._malloc(8);
        let pRANK = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pRCOND, RCOND, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        let pJPVT = emlapack._malloc(4 * (N));
        let aJPVT = new Int32Array(emlapack.HEAPI32.buffer, pJPVT, (N));
        aJPVT.set(JPVT.data, JPVT.offset);

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pM, pN, pNRHS, pA, pLDA, pB, pLDB, pJPVT, pRCOND, pRANK, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgelsy': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM, pN, pNRHS, pA, pLDA, pB, pLDB, pJPVT, pRCOND, pRANK, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgelsy': " + INFO);
        }

        return emlapack.getValue(pRANK, 'i32');
}