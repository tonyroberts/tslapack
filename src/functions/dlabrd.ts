
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlabrd(M: number,
    N: number,
    NB: number,
    A: ndarray<number>,
    LDA: number,
    LDX: number,
    LDY: number): ndarray<number> {

        let func = emlapack.cwrap('dlabrd_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] NB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] D: DOUBLE PRECISION[NB]
            'number', // [out] E: DOUBLE PRECISION[NB]
            'number', // [out] TAUQ: DOUBLE PRECISION[NB]
            'number', // [out] TAUP: DOUBLE PRECISION[NB]
            'number', // [out] X: DOUBLE PRECISION[LDX,NB]
            'number', // [in] LDX: INTEGER
            'number', // [out] Y: DOUBLE PRECISION[LDY,NB]
            'number', // [in] LDY: INTEGER
        ]);

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pNB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDX = emlapack._malloc(4);
        let pLDY = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNB, NB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDX, LDX, 'i32');
        emlapack.setValue(pLDY, LDY, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pD = emlapack._malloc(8 * (NB));
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (NB));

        let pE = emlapack._malloc(8 * (NB));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (NB));

        let pTAUQ = emlapack._malloc(8 * (NB));
        let TAUQ = new Float64Array(emlapack.HEAPF64.buffer, pTAUQ, (NB));

        let pTAUP = emlapack._malloc(8 * (NB));
        let TAUP = new Float64Array(emlapack.HEAPF64.buffer, pTAUP, (NB));

        let pX = emlapack._malloc(8 * (LDX) * (NB));
        let X = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (NB));

        let pY = emlapack._malloc(8 * (LDY) * (NB));
        let Y = new Float64Array(emlapack.HEAPF64.buffer, pY, (LDY) * (NB));

        func(pM, pN, pNB, pA, pLDA, pD, pE, pTAUQ, pTAUP, pX, pLDX, pY, pLDY);
        return ndarray(Y, [(LDY), (NB)]);
}