
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dorg2r(M: number,
    N: number,
    K: number,
    A: ndarray<number>,
    LDA: number,
    TAU: ndarray<number>) {

        let func = emlapack.cwrap('dorg2r_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] K: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] TAU: DOUBLE PRECISION[K]
            'number', // [out] WORK: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pK = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pK, K, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pTAU = emlapack._malloc(8 * (K));
        let aTAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (K));
        aTAU.set(TAU.data, TAU.offset);

        let pWORK = emlapack._malloc(8 * (N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N));

        func(pM, pN, pK, pA, pLDA, pTAU, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorg2r': " + INFO);
        }
}