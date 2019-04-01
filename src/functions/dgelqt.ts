
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgelqt(M: number,
    N: number,
    MB: number,
    A: ndarray<number>,
    LDA: number,
    LDT: number): ndarray<number> {

        let func = emlapack.cwrap('dgelqt_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] MB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] T: DOUBLE PRECISION[LDT,MIN(M,N)]
            'number', // [in] LDT: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MB*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pMB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pMB, MB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDT, LDT, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pT = emlapack._malloc(8 * (LDT) * (Math.min(M,N)));
        let T = new Float64Array(emlapack.HEAPF64.buffer, pT, (LDT) * (Math.min(M,N)));

        let pWORK = emlapack._malloc(8 * (MB*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (MB*N));

        func(pM, pN, pMB, pA, pLDA, pT, pLDT, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgelqt': " + INFO);
        }
        return ndarray(T, [(LDT), (Math.min(M,N))]);
}