
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dggev3(JOBVL: string,
    JOBVR: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    LDVL: number,
    LDVR: number): ndarray<number> {

        let func = emlapack.cwrap('dggev3_', null, [
            'number', // [in] JOBVL: CHARACTER
            'number', // [in] JOBVR: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA, N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB, N]
            'number', // [in] LDB: INTEGER
            'number', // [out] ALPHAR: DOUBLE PRECISION[N]
            'number', // [out] ALPHAI: DOUBLE PRECISION[N]
            'number', // [out] BETA: DOUBLE PRECISION[N]
            'number', // [out] VL: DOUBLE PRECISION[LDVL,N]
            'number', // [in] LDVL: INTEGER
            'number', // [out] VR: DOUBLE PRECISION[LDVR,N]
            'number', // [in] LDVR: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOBVL = emlapack._malloc(1);
        let pJOBVR = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDVL = emlapack._malloc(4);
        let pLDVR = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBVL, JOBVL.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBVR, JOBVR.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDVL, LDVL, 'i32');
        emlapack.setValue(pLDVR, LDVR, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pALPHAR = emlapack._malloc(8 * (N));
        let ALPHAR = new Float64Array(emlapack.HEAPF64.buffer, pALPHAR, (N));

        let pALPHAI = emlapack._malloc(8 * (N));
        let ALPHAI = new Float64Array(emlapack.HEAPF64.buffer, pALPHAI, (N));

        let pBETA = emlapack._malloc(8 * (N));
        let BETA = new Float64Array(emlapack.HEAPF64.buffer, pBETA, (N));

        let pVL = emlapack._malloc(8 * (LDVL) * (N));
        let VL = new Float64Array(emlapack.HEAPF64.buffer, pVL, (LDVL) * (N));

        let pVR = emlapack._malloc(8 * (LDVR) * (N));
        let VR = new Float64Array(emlapack.HEAPF64.buffer, pVR, (LDVR) * (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pJOBVL, pJOBVR, pN, pA, pLDA, pB, pLDB, pALPHAR, pALPHAI, pBETA, pVL, pLDVL, pVR, pLDVR, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggev3': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOBVL, pJOBVR, pN, pA, pLDA, pB, pLDB, pALPHAR, pALPHAI, pBETA, pVL, pLDVL, pVR, pLDVR, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggev3': " + INFO);
        }

        return ndarray(VR, [(LDVR), (N)]);
}