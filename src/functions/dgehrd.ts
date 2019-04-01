
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgehrd(N: number,
    ILO: number,
    IHI: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dgehrd_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] ILO: INTEGER
            'number', // [in] IHI: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] TAU: DOUBLE PRECISION[N-1]
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pN = emlapack._malloc(4);
        let pILO = emlapack._malloc(4);
        let pIHI = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pILO, ILO, 'i32');
        emlapack.setValue(pIHI, IHI, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pTAU = emlapack._malloc(8 * (N-1));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (N-1));

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        func(pN, pILO, pIHI, pA, pLDA, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgehrd': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pN, pILO, pIHI, pA, pLDA, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgehrd': " + INFO);
        }

        return ndarray(TAU, [(N-1)]);
}