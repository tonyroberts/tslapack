
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasyf(UPLO: string,
    N: number,
    NB: number,
    A: ndarray<number>,
    LDA: number,
    LDW: number): ndarray<number> {

        let func = emlapack.cwrap('dlasyf_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NB: INTEGER
            'number', // [out] KB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] IPIV: INTEGER[N]
            'number', // [out] W: DOUBLE PRECISION[LDW,NB]
            'number', // [in] LDW: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNB = emlapack._malloc(4);
        let pKB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDW = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNB, NB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDW, LDW, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let IPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));

        let pW = emlapack._malloc(8 * (LDW) * (NB));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (LDW) * (NB));

        func(pUPLO, pN, pNB, pKB, pA, pLDA, pIPIV, pW, pLDW, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlasyf': " + INFO);
        }
        return ndarray(W, [(LDW), (NB)]);
}