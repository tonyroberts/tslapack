
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dggrqf(M: number,
    P: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number): ndarray<number> {

        let func = emlapack.cwrap('dggrqf_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] TAUA: DOUBLE PRECISION[min(M,N)]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [out] TAUB: DOUBLE PRECISION[min(P,N)]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pTAUA = emlapack._malloc(8 * (Math.min(M,N)));
        let TAUA = new Float64Array(emlapack.HEAPF64.buffer, pTAUA, (Math.min(M,N)));

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pTAUB = emlapack._malloc(8 * (Math.min(P,N)));
        let TAUB = new Float64Array(emlapack.HEAPF64.buffer, pTAUB, (Math.min(P,N)));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pM, pP, pN, pA, pLDA, pTAUA, pB, pLDB, pTAUB, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggrqf': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM, pP, pN, pA, pLDA, pTAUA, pB, pLDB, pTAUB, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dggrqf': " + INFO);
        }

        return ndarray(TAUB, [(Math.min(P,N))]);
}