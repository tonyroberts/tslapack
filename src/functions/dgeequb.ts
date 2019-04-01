
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgeequb(M: number,
    N: number,
    A: ndarray<number>,
    LDA: number): number {

        let func = emlapack.cwrap('dgeequb_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] R: DOUBLE PRECISION[M]
            'number', // [out] C: DOUBLE PRECISION[N]
            'number', // [out] ROWCND: DOUBLE PRECISION
            'number', // [out] COLCND: DOUBLE PRECISION
            'number', // [out] AMAX: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pROWCND = emlapack._malloc(8);
        let pCOLCND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pR = emlapack._malloc(8 * (M));
        let R = new Float64Array(emlapack.HEAPF64.buffer, pR, (M));

        let pC = emlapack._malloc(8 * (N));
        let C = new Float64Array(emlapack.HEAPF64.buffer, pC, (N));

        func(pM, pN, pA, pLDA, pR, pC, pROWCND, pCOLCND, pAMAX, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgeequb': " + INFO);
        }
        return emlapack.getValue(pAMAX, 'double');
}