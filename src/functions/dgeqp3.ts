
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgeqp3(M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    JPVT: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dgeqp3_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] JPVT: INTEGER[N]
            'number', // [out] TAU: DOUBLE PRECISION[min(M,N)]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pJPVT = emlapack._malloc(4 * (N));
        let aJPVT = new Int32Array(emlapack.HEAPI32.buffer, pJPVT, (N));
        aJPVT.set(JPVT.data, JPVT.offset);

        let pTAU = emlapack._malloc(8 * (Math.min(M,N)));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (Math.min(M,N)));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pM, pN, pA, pLDA, pJPVT, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgeqp3': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM, pN, pA, pLDA, pJPVT, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgeqp3': " + INFO);
        }

        return ndarray(TAU, [(Math.min(M,N))]);
}