
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtgsja(JOBU: string,
    JOBV: string,
    JOBQ: string,
    M: number,
    P: number,
    N: number,
    K: number,
    L: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    TOLA: number,
    TOLB: number,
    U: ndarray<number>,
    LDU: number,
    V: ndarray<number>,
    LDV: number,
    Q: ndarray<number>,
    LDQ: number): number {

        let func = emlapack.cwrap('dtgsja_', null, [
            'number', // [in] JOBU: CHARACTER
            'number', // [in] JOBV: CHARACTER
            'number', // [in] JOBQ: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] K: INTEGER
            'number', // [in] L: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [in] TOLA: DOUBLE PRECISION
            'number', // [in] TOLB: DOUBLE PRECISION
            'number', // [out] ALPHA: DOUBLE PRECISION[N]
            'number', // [out] BETA: DOUBLE PRECISION[N]
            'number', // [in,out] U: DOUBLE PRECISION[LDU,M]
            'number', // [in] LDU: INTEGER
            'number', // [in,out] V: DOUBLE PRECISION[LDV,P]
            'number', // [in] LDV: INTEGER
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] NCYCLE: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOBU = emlapack._malloc(1);
        let pJOBV = emlapack._malloc(1);
        let pJOBQ = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pK = emlapack._malloc(4);
        let pL = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pTOLA = emlapack._malloc(8);
        let pTOLB = emlapack._malloc(8);
        let pLDU = emlapack._malloc(4);
        let pLDV = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pNCYCLE = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBU, JOBU.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBV, JOBV.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBQ, JOBQ.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pK, K, 'i32');
        emlapack.setValue(pL, L, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pTOLA, TOLA, 'double');
        emlapack.setValue(pTOLB, TOLB, 'double');
        emlapack.setValue(pLDU, LDU, 'i32');
        emlapack.setValue(pLDV, LDV, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');

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
        let aU = new Float64Array(emlapack.HEAPF64.buffer, pU, (LDU) * (M));
        aU.set(U.data, U.offset);

        let pV = emlapack._malloc(8 * (LDV) * (P));
        let aV = new Float64Array(emlapack.HEAPF64.buffer, pV, (LDV) * (P));
        aV.set(V.data, V.offset);

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let aQ = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));
        aQ.set(Q.data, Q.offset);

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        func(pJOBU, pJOBV, pJOBQ, pM, pP, pN, pK, pL, pA, pLDA, pB, pLDB, pTOLA, pTOLB, pALPHA, pBETA, pU, pLDU, pV, pLDV, pQ, pLDQ, pWORK, pNCYCLE, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtgsja': " + INFO);
        }
        return emlapack.getValue(pNCYCLE, 'i32');
}