
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpoequb(N: number,
    A: ndarray<number>,
    LDA: number): number {

        let func = emlapack.cwrap('dpoequb_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] S: DOUBLE PRECISION[N]
            'number', // [out] SCOND: DOUBLE PRECISION
            'number', // [out] AMAX: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pSCOND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pS = emlapack._malloc(8 * (N));
        let S = new Float64Array(emlapack.HEAPF64.buffer, pS, (N));

        func(pN, pA, pLDA, pS, pSCOND, pAMAX, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpoequb': " + INFO);
        }
        return emlapack.getValue(pAMAX, 'double');
}