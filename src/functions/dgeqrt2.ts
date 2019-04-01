
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgeqrt2(M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    LDT: number): ndarray<number> {

        let func = emlapack.cwrap('dgeqrt2_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] T: DOUBLE PRECISION[LDT,N]
            'number', // [in] LDT: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDT, LDT, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pT = emlapack._malloc(8 * (LDT) * (N));
        let T = new Float64Array(emlapack.HEAPF64.buffer, pT, (LDT) * (N));

        func(pM, pN, pA, pLDA, pT, pLDT, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgeqrt2': " + INFO);
        }
        return ndarray(T, [(LDT), (N)]);
}