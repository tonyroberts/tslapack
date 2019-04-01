
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgebrd(M: number,
    N: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dgebrd_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] D: DOUBLE PRECISION[min(M,N)]
            'number', // [out] E: DOUBLE PRECISION[min(M,N)-1]
            'number', // [out] TAUQ: DOUBLE PRECISION[min(M,N)]
            'number', // [out] TAUP: DOUBLE PRECISION[min(M,N)]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pD = emlapack._malloc(8 * (Math.min(M,N)));
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (Math.min(M,N)));

        let pE = emlapack._malloc(8 * (Math.min(M,N)-1));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (Math.min(M,N)-1));

        let pTAUQ = emlapack._malloc(8 * (Math.min(M,N)));
        let TAUQ = new Float64Array(emlapack.HEAPF64.buffer, pTAUQ, (Math.min(M,N)));

        let pTAUP = emlapack._malloc(8 * (Math.min(M,N)));
        let TAUP = new Float64Array(emlapack.HEAPF64.buffer, pTAUP, (Math.min(M,N)));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pM, pN, pA, pLDA, pD, pE, pTAUQ, pTAUP, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgebrd': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM, pN, pA, pLDA, pD, pE, pTAUQ, pTAUP, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgebrd': " + INFO);
        }

        return ndarray(TAUP, [(Math.min(M,N))]);
}