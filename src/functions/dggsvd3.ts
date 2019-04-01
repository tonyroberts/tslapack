
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dggsvd3(JOBU: string,
    JOBV: string,
    JOBQ: string,
    M: number,
    N: number,
    P: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    LDU: number,
    LDV: number,
    LDQ: number): ndarray<number> {

        let func = emlapack.cwrap('dggsvd3_', null, [
            'number', // [in] JOBU: CHARACTER
            'number', // [in] JOBV: CHARACTER
            'number', // [in] JOBQ: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] P: INTEGER
            'number', // [out] K: INTEGER
            'number', // [out] L: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [out] ALPHA: DOUBLE PRECISION[N]
            'number', // [out] BETA: DOUBLE PRECISION[N]
            'number', // [out] U: DOUBLE PRECISION[LDU,M]
            'number', // [in] LDU: INTEGER
            'number', // [out] V: DOUBLE PRECISION[LDV,P]
            'number', // [in] LDV: INTEGER
            'number', // [out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOBU = emlapack._malloc(1);
        let pJOBV = emlapack._malloc(1);
        let pJOBQ = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pK = emlapack._malloc(4);
        let pL = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDU = emlapack._malloc(4);
        let pLDV = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBU, JOBU.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBV, JOBV.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBQ, JOBQ.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDU, LDU, 'i32');
        emlapack.setValue(pLDV, LDV, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pALPHA = emlapack._malloc(8 * (N));
        let ALPHA = new Float64Array(emlapack.HEAPF64.buffer, pALPHA, (N));

        let pBETA = emlapack._malloc(8 * (N));
        let BETA = new Float64Array(emlapack.HEAPF64.buffer, pBETA, (N));

        let pU = emlapack._malloc(8 * (LDU) * (M));
        let U = new Float64Array(emlapack.HEAPF64.buffer, pU, (LDU) * (M));

        let pV = emlapack._malloc(8 * (LDV) * (P));
        let V = new Float64Array(emlapack.HEAPF64.buffer, pV, (LDV) * (P));

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let Q = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pJOBU, pJOBV, pJOBQ, pM, pN, pP, pK, pL, pA, pLDA, pB, pLDB, pALPHA, pBETA, pU, pLDU, pV, pLDV, pQ, pLDQ, pWORK, pLWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggsvd3': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOBU, pJOBV, pJOBQ, pM, pN, pP, pK, pL, pA, pLDA, pB, pLDB, pALPHA, pBETA, pU, pLDU, pV, pLDV, pQ, pLDQ, pWORK, pLWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggsvd3': " + INFO);
        }

        return ndarray(Q, [(LDQ), (N)]);
}