
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgetc2(N: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dgetc2_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA, N]
            'number', // [in] LDA: INTEGER
            'number', // [out] IPIV: INTEGER[N]
            'number', // [out] JPIV: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let IPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));

        let pJPIV = emlapack._malloc(4 * (N));
        let JPIV = new Int32Array(emlapack.HEAPI32.buffer, pJPIV, (N));

        func(pN, pA, pLDA, pIPIV, pJPIV, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgetc2': " + INFO);
        }
        return ndarray(JPIV, [(N)]);
}