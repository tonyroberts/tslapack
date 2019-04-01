
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgesvj(JOBA: string,
    JOBU: string,
    JOBV: string,
    M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    MV: number,
    V: ndarray<number>,
    LDV: number,
    WORK: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dgesvj_', null, [
            'number', // [in] JOBA: CHARACTER
            'number', // [in] JOBU: CHARACTER
            'number', // [in] JOBV: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] SVA: DOUBLE PRECISION[N]
            'number', // [in] MV: INTEGER
            'number', // [in,out] V: DOUBLE PRECISION[LDV,N]
            'number', // [in] LDV: INTEGER
            'number', // [in,out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOBA = emlapack._malloc(1);
        let pJOBU = emlapack._malloc(1);
        let pJOBV = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pMV = emlapack._malloc(4);
        let pLDV = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBA, JOBA.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBU, JOBU.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBV, JOBV.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pMV, MV, 'i32');
        emlapack.setValue(pLDV, LDV, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pSVA = emlapack._malloc(8 * (N));
        let SVA = new Float64Array(emlapack.HEAPF64.buffer, pSVA, (N));

        let pV = emlapack._malloc(8 * (LDV) * (N));
        let aV = new Float64Array(emlapack.HEAPF64.buffer, pV, (LDV) * (N));
        aV.set(V.data, V.offset);

        let pWORK = emlapack._malloc(8 * (LWORK));
        let aWORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));
        aWORK.set(WORK.data, WORK.offset);

        func(pJOBA, pJOBU, pJOBV, pM, pN, pA, pLDA, pSVA, pMV, pV, pLDV, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgesvj': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOBA, pJOBU, pJOBV, pM, pN, pA, pLDA, pSVA, pMV, pV, pLDV, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgesvj': " + INFO);
        }

        return ndarray(SVA, [(N)]);
}