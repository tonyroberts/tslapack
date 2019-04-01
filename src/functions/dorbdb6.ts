
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dorbdb6(M1: number,
    M2: number,
    N: number,
    X1: ndarray<number>,
    INCX1: number,
    X2: ndarray<number>,
    INCX2: number,
    Q1: ndarray<number>,
    LDQ1: number,
    Q2: ndarray<number>,
    LDQ2: number) {

        let func = emlapack.cwrap('dorbdb6_', null, [
            'number', // [in] M1: INTEGER
            'number', // [in] M2: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] X1: DOUBLE PRECISION[M1]
            'number', // [in] INCX1: INTEGER
            'number', // [in,out] X2: DOUBLE PRECISION[M2]
            'number', // [in] INCX2: INTEGER
            'number', // [in] Q1: DOUBLE PRECISION[LDQ1, N]
            'number', // [in] LDQ1: INTEGER
            'number', // [in] Q2: DOUBLE PRECISION[LDQ2, N]
            'number', // [in] LDQ2: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM1 = emlapack._malloc(4);
        let pM2 = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pINCX1 = emlapack._malloc(4);
        let pINCX2 = emlapack._malloc(4);
        let pLDQ1 = emlapack._malloc(4);
        let pLDQ2 = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM1, M1, 'i32');
        emlapack.setValue(pM2, M2, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pINCX1, INCX1, 'i32');
        emlapack.setValue(pINCX2, INCX2, 'i32');
        emlapack.setValue(pLDQ1, LDQ1, 'i32');
        emlapack.setValue(pLDQ2, LDQ2, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pX1 = emlapack._malloc(8 * (M1));
        let aX1 = new Float64Array(emlapack.HEAPF64.buffer, pX1, (M1));
        aX1.set(X1.data, X1.offset);

        let pX2 = emlapack._malloc(8 * (M2));
        let aX2 = new Float64Array(emlapack.HEAPF64.buffer, pX2, (M2));
        aX2.set(X2.data, X2.offset);

        let pQ1 = emlapack._malloc(8 * (LDQ1) * (N));
        let aQ1 = new Float64Array(emlapack.HEAPF64.buffer, pQ1, (LDQ1) * (N));
        aQ1.set(Q1.data, Q1.offset);

        let pQ2 = emlapack._malloc(8 * (LDQ2) * (N));
        let aQ2 = new Float64Array(emlapack.HEAPF64.buffer, pQ2, (LDQ2) * (N));
        aQ2.set(Q2.data, Q2.offset);

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        func(pM1, pM2, pN, pX1, pINCX1, pX2, pINCX2, pQ1, pLDQ1, pQ2, pLDQ2, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorbdb6': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM1, pM2, pN, pX1, pINCX1, pX2, pINCX2, pQ1, pLDQ1, pQ2, pLDQ2, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorbdb6': " + INFO);
        }

}