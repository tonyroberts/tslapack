
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtpttr(UPLO: string,
    N: number,
    AP: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dtpttr_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] A: DOUBLE PRECISION[LDA, N]
            'number', // [in] LDA: INTEGER
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

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let A = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));

        func(pUPLO, pN, pAP, pA, pLDA, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtpttr': " + INFO);
        }
        return ndarray(A, [(LDA), (N)]);
}