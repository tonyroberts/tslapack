
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgglse(M: number,
    N: number,
    P: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    C: ndarray<number>,
    D: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dgglse_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [in,out] C: DOUBLE PRECISION[M]
            'number', // [in,out] D: DOUBLE PRECISION[P]
            'number', // [out] X: DOUBLE PRECISION[N]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pC = emlapack._malloc(8 * (M));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (M));
        aC.set(C.data, C.offset);

        let pD = emlapack._malloc(8 * (P));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (P));
        aD.set(D.data, D.offset);

        let pX = emlapack._malloc(8 * (N));
        let X = new Float64Array(emlapack.HEAPF64.buffer, pX, (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pM, pN, pP, pA, pLDA, pB, pLDB, pC, pD, pX, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgglse': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM, pN, pP, pA, pLDA, pB, pLDB, pC, pD, pX, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgglse': " + INFO);
        }

        return ndarray(X, [(N)]);
}