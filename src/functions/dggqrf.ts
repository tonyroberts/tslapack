
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dggqrf(N: number,
    M: number,
    P: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number): ndarray<number> {

        let func = emlapack.cwrap('dggqrf_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] M: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,M]
            'number', // [in] LDA: INTEGER
            'number', // [out] TAUA: DOUBLE PRECISION[min(N,M)]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,P]
            'number', // [in] LDB: INTEGER
            'number', // [out] TAUB: DOUBLE PRECISION[min(N,P)]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pN = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (M));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (M));
        aA.set(A.data, A.offset);

        let pTAUA = emlapack._malloc(8 * (Math.min(N,M)));
        let TAUA = new Float64Array(emlapack.HEAPF64.buffer, pTAUA, (Math.min(N,M)));

        let pB = emlapack._malloc(8 * (LDB) * (P));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (P));
        aB.set(B.data, B.offset);

        let pTAUB = emlapack._malloc(8 * (Math.min(N,P)));
        let TAUB = new Float64Array(emlapack.HEAPF64.buffer, pTAUB, (Math.min(N,P)));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pN, pM, pP, pA, pLDA, pTAUA, pB, pLDB, pTAUB, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggqrf': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pN, pM, pP, pA, pLDA, pTAUA, pB, pLDB, pTAUB, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggqrf': " + INFO);
        }

        return ndarray(TAUB, [(Math.min(N,P))]);
}