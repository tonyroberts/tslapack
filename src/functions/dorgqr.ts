
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dorgqr(M: number,
    N: number,
    K: number,
    A: ndarray<number>,
    LDA: number,
    TAU: ndarray<number>) {

        let func = emlapack.cwrap('dorgqr_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] K: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] TAU: DOUBLE PRECISION[K]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pK = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pK, K, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pTAU = emlapack._malloc(8 * (K));
        let aTAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (K));
        aTAU.set(TAU.data, TAU.offset);

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pM, pN, pK, pA, pLDA, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorgqr': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM, pN, pK, pA, pLDA, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorgqr': " + INFO);
        }

}