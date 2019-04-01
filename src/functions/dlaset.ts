
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaset(UPLO: string,
    M: number,
    N: number,
    ALPHA: number,
    BETA: number,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dlaset_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] ALPHA: DOUBLE PRECISION
            'number', // [in] BETA: DOUBLE PRECISION
            'number', // [out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
        ]);

        let pUPLO = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pALPHA = emlapack._malloc(8);
        let pBETA = emlapack._malloc(8);
        let pLDA = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pALPHA, ALPHA, 'double');
        emlapack.setValue(pBETA, BETA, 'double');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let A = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));

        func(pUPLO, pM, pN, pALPHA, pBETA, pA, pLDA);
        return ndarray(A, [(LDA), (N)]);
}