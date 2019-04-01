
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgejsv(JOBA: string,
    JOBU: string,
    JOBV: string,
    JOBR: string,
    JOBT: string,
    JOBP: string,
    M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    LDU: number,
    LDV: number): ndarray<number> {

        let func = emlapack.cwrap('dgejsv_', null, [
            'number', // [in] JOBA: CHARACTER
            'number', // [in] JOBU: CHARACTER
            'number', // [in] JOBV: CHARACTER
            'number', // [in] JOBR: CHARACTER
            'number', // [in] JOBT: CHARACTER
            'number', // [in] JOBP: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] SVA: DOUBLE PRECISION[N]
            'number', // [out] U: DOUBLE PRECISION[LDU, N]
            'number', // [in] LDU: INTEGER
            'number', // [out] V: DOUBLE PRECISION[LDV, N]
            'number', // [in] LDV: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] IWORK: INTEGER[M+3*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOBA = emlapack._malloc(1);
        let pJOBU = emlapack._malloc(1);
        let pJOBV = emlapack._malloc(1);
        let pJOBR = emlapack._malloc(1);
        let pJOBT = emlapack._malloc(1);
        let pJOBP = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDU = emlapack._malloc(4);
        let pLDV = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBA, JOBA.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBU, JOBU.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBV, JOBV.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBR, JOBR.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBT, JOBT.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBP, JOBP.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDU, LDU, 'i32');
        emlapack.setValue(pLDV, LDV, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pSVA = emlapack._malloc(8 * (N));
        let SVA = new Float64Array(emlapack.HEAPF64.buffer, pSVA, (N));

        let pU = emlapack._malloc(8 * (LDU) * (N));
        let U = new Float64Array(emlapack.HEAPF64.buffer, pU, (LDU) * (N));

        let pV = emlapack._malloc(8 * (LDV) * (N));
        let V = new Float64Array(emlapack.HEAPF64.buffer, pV, (LDV) * (N));

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        let pIWORK = emlapack._malloc(4 * (M+3*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (M+3*N));

        func(pJOBA, pJOBU, pJOBV, pJOBR, pJOBT, pJOBP, pM, pN, pA, pLDA, pSVA, pU, pLDU, pV, pLDV, pWORK, pLWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgejsv': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOBA, pJOBU, pJOBV, pJOBR, pJOBT, pJOBP, pM, pN, pA, pLDA, pSVA, pU, pLDU, pV, pLDV, pWORK, pLWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgejsv': " + INFO);
        }

        return ndarray(V, [(LDV), (N)]);
}