
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgeqrt(M: number,
    N: number,
    NB: number,
    A: ndarray<number>,
    LDA: number,
    LDT: number): ndarray<number> {

        let func = emlapack.cwrap('dgeqrt_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] NB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] T: DOUBLE PRECISION[LDT,MIN(M,N)]
            'number', // [in] LDT: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[NB*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pNB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNB, NB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDT, LDT, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pT = emlapack._malloc(8 * (LDT) * (Math.min(M,N)));
        let T = new Float64Array(emlapack.HEAPF64.buffer, pT, (LDT) * (Math.min(M,N)));

        let pWORK = emlapack._malloc(8 * (NB*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (NB*N));

        func(pM, pN, pNB, pA, pLDA, pT, pLDT, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgeqrt': " + INFO);
        }
        return ndarray(T, [(LDT), (Math.min(M,N))]);
}