
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgetf2(M: number,
    N: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dgetf2_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] IPIV: INTEGER[min(M,N)]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pIPIV = emlapack._malloc(4 * (Math.min(M,N)));
        let IPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (Math.min(M,N)));

        func(pM, pN, pA, pLDA, pIPIV, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgetf2': " + INFO);
        }
        return ndarray(IPIV, [(Math.min(M,N))]);
}