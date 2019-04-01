
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlahr2(N: number,
    K: number,
    NB: number,
    A: ndarray<number>,
    LDA: number,
    LDT: number,
    LDY: number): ndarray<number> {

        let func = emlapack.cwrap('dlahr2_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] K: INTEGER
            'number', // [in] NB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N-K+1]
            'number', // [in] LDA: INTEGER
            'number', // [out] TAU: DOUBLE PRECISION[NB]
            'number', // [out] T: DOUBLE PRECISION[LDT,NB]
            'number', // [in] LDT: INTEGER
            'number', // [out] Y: DOUBLE PRECISION[LDY,NB]
            'number', // [in] LDY: INTEGER
        ]);

        let pN = emlapack._malloc(4);
        let pK = emlapack._malloc(4);
        let pNB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDT = emlapack._malloc(4);
        let pLDY = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pK, K, 'i32');
        emlapack.setValue(pNB, NB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDT, LDT, 'i32');
        emlapack.setValue(pLDY, LDY, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N-K+1));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N-K+1));
        aA.set(A.data, A.offset);

        let pTAU = emlapack._malloc(8 * (NB));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (NB));

        let pT = emlapack._malloc(8 * (LDT) * (NB));
        let T = new Float64Array(emlapack.HEAPF64.buffer, pT, (LDT) * (NB));

        let pY = emlapack._malloc(8 * (LDY) * (NB));
        let Y = new Float64Array(emlapack.HEAPF64.buffer, pY, (LDY) * (NB));

        func(pN, pK, pNB, pA, pLDA, pTAU, pT, pLDT, pY, pLDY);
        return ndarray(Y, [(LDY), (NB)]);
}