
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlatrz(M: number,
    N: number,
    L: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dlatrz_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] L: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] TAU: DOUBLE PRECISION[M]
            'number', // [out] WORK: DOUBLE PRECISION[M]
        ]);

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pL = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pL, L, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pTAU = emlapack._malloc(8 * (M));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (M));

        let pWORK = emlapack._malloc(8 * (M));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (M));

        func(pM, pN, pL, pA, pLDA, pTAU, pWORK);
        return ndarray(TAU, [(M)]);
}