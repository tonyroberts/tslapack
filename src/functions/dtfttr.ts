
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtfttr(TRANSR: string,
    UPLO: string,
    N: number,
    ARF: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dtfttr_', null, [
            'number', // [in] TRANSR: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] ARF: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pTRANSR = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANSR, TRANSR.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pARF = emlapack._malloc(8 * (N*(N+1)/2));
        let aARF = new Float64Array(emlapack.HEAPF64.buffer, pARF, (N*(N+1)/2));
        aARF.set(ARF.data, ARF.offset);

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let A = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));

        func(pTRANSR, pUPLO, pN, pARF, pA, pLDA, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtfttr': " + INFO);
        }
        return ndarray(A, [(LDA), (N)]);
}