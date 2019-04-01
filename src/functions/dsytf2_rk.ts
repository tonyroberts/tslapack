
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsytf2_rk(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dsytf2_rk_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] E: DOUBLE PRECISION[N]
            'number', // [out] IPIV: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pE = emlapack._malloc(8 * (N));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (N));

        let pIPIV = emlapack._malloc(4 * (N));
        let IPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));

        func(pUPLO, pN, pA, pLDA, pE, pIPIV, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsytf2_rk': " + INFO);
        }
        return ndarray(IPIV, [(N)]);
}