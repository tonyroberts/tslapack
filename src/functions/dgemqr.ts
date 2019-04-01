
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgemqr(SIDE: string,
    TRANS: string,
    M: number,
    N: number,
    K: number,
    A: ndarray<number>,
    LDA: number,
    T: ndarray<number>,
    TSIZE: number,
    C: ndarray<number>,
    LDC: number) {

        let func = emlapack.cwrap('dgemqr_', null, [
            'number', // [in] SIDE: CHARACTER
            'number', // [in] TRANS: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] K: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,K]
            'number', // [in] LDA: INTEGER
            'number', // [in] T: DOUBLE PRECISION[MAX(5,TSIZE)]
            'number', // [in] TSIZE: INTEGER
            'number', // [in,out] C: DOUBLE PRECISION[LDC,N]
            'number', // [in] LDC: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pSIDE = emlapack._malloc(1);
        let pTRANS = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pK = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pTSIZE = emlapack._malloc(4);
        let pLDC = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pSIDE, SIDE.charCodeAt(0), 'i8');
        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pK, K, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pTSIZE, TSIZE, 'i32');
        emlapack.setValue(pLDC, LDC, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (K));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (K));
        aA.set(A.data, A.offset);

        let pT = emlapack._malloc(8 * (Math.max(5,TSIZE)));
        let aT = new Float64Array(emlapack.HEAPF64.buffer, pT, (Math.max(5,TSIZE)));
        aT.set(T.data, T.offset);

        let pC = emlapack._malloc(8 * (LDC) * (N));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (LDC) * (N));
        aC.set(C.data, C.offset);

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pSIDE, pTRANS, pM, pN, pK, pA, pLDA, pT, pTSIZE, pC, pLDC, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgemqr': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pSIDE, pTRANS, pM, pN, pK, pA, pLDA, pT, pTSIZE, pC, pLDC, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgemqr': " + INFO);
        }

}