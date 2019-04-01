
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgsvj0(JOBV: string,
    M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    D: ndarray<number>,
    SVA: ndarray<number>,
    MV: number,
    V: ndarray<number>,
    LDV: number,
    EPS: number,
    SFMIN: number,
    TOL: number,
    NSWEEP: number) {

        let func = emlapack.cwrap('dgsvj0_', null, [
            'number', // [in] JOBV: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] SVA: DOUBLE PRECISION[N]
            'number', // [in] MV: INTEGER
            'number', // [in,out] V: DOUBLE PRECISION[LDV,N]
            'number', // [in] LDV: INTEGER
            'number', // [in] EPS: DOUBLE PRECISION
            'number', // [in] SFMIN: DOUBLE PRECISION
            'number', // [in] TOL: DOUBLE PRECISION
            'number', // [in] NSWEEP: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOBV = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pMV = emlapack._malloc(4);
        let pLDV = emlapack._malloc(4);
        let pEPS = emlapack._malloc(8);
        let pSFMIN = emlapack._malloc(8);
        let pTOL = emlapack._malloc(8);
        let pNSWEEP = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBV, JOBV.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pMV, MV, 'i32');
        emlapack.setValue(pLDV, LDV, 'i32');
        emlapack.setValue(pEPS, EPS, 'double');
        emlapack.setValue(pSFMIN, SFMIN, 'double');
        emlapack.setValue(pTOL, TOL, 'double');
        emlapack.setValue(pNSWEEP, NSWEEP, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pSVA = emlapack._malloc(8 * (N));
        let aSVA = new Float64Array(emlapack.HEAPF64.buffer, pSVA, (N));
        aSVA.set(SVA.data, SVA.offset);

        let pV = emlapack._malloc(8 * (LDV) * (N));
        let aV = new Float64Array(emlapack.HEAPF64.buffer, pV, (LDV) * (N));
        aV.set(V.data, V.offset);

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        func(pJOBV, pM, pN, pA, pLDA, pD, pSVA, pMV, pV, pLDV, pEPS, pSFMIN, pTOL, pNSWEEP, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgsvj0': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOBV, pM, pN, pA, pLDA, pD, pSVA, pMV, pV, pLDV, pEPS, pSFMIN, pTOL, pNSWEEP, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgsvj0': " + INFO);
        }

}