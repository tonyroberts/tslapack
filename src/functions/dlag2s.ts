
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlag2s(M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    LDSA: number): ndarray<number> {

        let func = emlapack.cwrap('dlag2s_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] SA: REAL[LDSA,N]
            'number', // [in] LDSA: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDSA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDSA, LDSA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pSA = emlapack._malloc(8 * (LDSA) * (N));
        let SA = new Float64Array(emlapack.HEAPF64.buffer, pSA, (LDSA) * (N));

        func(pM, pN, pA, pLDA, pSA, pLDSA, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlag2s': " + INFO);
        }
        return ndarray(SA, [(LDSA), (N)]);
}