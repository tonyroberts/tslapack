
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtrttp(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dtrttp_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] AP: DOUBLE PRECISION[N*(N+1)/2]
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

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let AP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));

        func(pUPLO, pN, pA, pLDA, pAP, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtrttp': " + INFO);
        }
        return ndarray(AP, [(N*(N+1)/2)]);
}