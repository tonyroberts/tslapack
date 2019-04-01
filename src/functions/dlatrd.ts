
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlatrd(UPLO: string,
    N: number,
    NB: number,
    A: ndarray<number>,
    LDA: number,
    LDW: number): ndarray<number> {

        let func = emlapack.cwrap('dlatrd_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] E: DOUBLE PRECISION[N-1]
            'number', // [out] TAU: DOUBLE PRECISION[N-1]
            'number', // [out] W: DOUBLE PRECISION[LDW,NB]
            'number', // [in] LDW: INTEGER
        ]);

        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDW = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNB, NB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDW, LDW, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));

        let pTAU = emlapack._malloc(8 * (N-1));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (N-1));

        let pW = emlapack._malloc(8 * (LDW) * (NB));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (LDW) * (NB));

        func(pUPLO, pN, pNB, pA, pLDA, pE, pTAU, pW, pLDW);
        return ndarray(W, [(LDW), (NB)]);
}