
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dggsvp3(JOBU: string,
    JOBV: string,
    JOBQ: string,
    M: number,
    P: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    TOLA: number,
    TOLB: number,
    LDU: number,
    LDV: number,
    LDQ: number): ndarray<number> {

        let func = emlapack.cwrap('dggsvp3_', null, [
            'number', // [in] JOBU: CHARACTER
            'number', // [in] JOBV: CHARACTER
            'number', // [in] JOBQ: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [in] TOLA: DOUBLE PRECISION
            'number', // [in] TOLB: DOUBLE PRECISION
            'number', // [out] K: INTEGER
            'number', // [out] L: INTEGER
            'number', // [out] U: DOUBLE PRECISION[LDU,M]
            'number', // [in] LDU: INTEGER
            'number', // [out] V: DOUBLE PRECISION[LDV,P]
            'number', // [in] LDV: INTEGER
            'number', // [out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] TAU: DOUBLE PRECISION[N]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOBU = emlapack._malloc(1);
        let pJOBV = emlapack._malloc(1);
        let pJOBQ = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pTOLA = emlapack._malloc(8);
        let pTOLB = emlapack._malloc(8);
        let pK = emlapack._malloc(4);
        let pL = emlapack._malloc(4);
        let pLDU = emlapack._malloc(4);
        let pLDV = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBU, JOBU.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBV, JOBV.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBQ, JOBQ.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pTOLA, TOLA, 'double');
        emlapack.setValue(pTOLB, TOLB, 'double');
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

        let pU = emlapack._malloc(8 * (LDU) * (M));
        let U = new Float64Array(emlapack.HEAPF64.buffer, pU, (LDU) * (M));

        let pV = emlapack._malloc(8 * (LDV) * (P));
        let V = new Float64Array(emlapack.HEAPF64.buffer, pV, (LDV) * (P));

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let Q = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        let pTAU = emlapack._malloc(8 * (N));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pJOBU, pJOBV, pJOBQ, pM, pP, pN, pA, pLDA, pB, pLDB, pTOLA, pTOLB, pK, pL, pU, pLDU, pV, pLDV, pQ, pLDQ, pIWORK, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggsvp3': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOBU, pJOBV, pJOBQ, pM, pP, pN, pA, pLDA, pB, pLDB, pTOLA, pTOLB, pK, pL, pU, pLDU, pV, pLDV, pQ, pLDQ, pIWORK, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggsvp3': " + INFO);
        }

        return ndarray(TAU, [(N)]);
}