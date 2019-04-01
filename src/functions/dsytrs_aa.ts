
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsytrs_aa(UPLO: string,
    N: number,
    NRHS: number,
    A: ndarray<number>,
    LDA: number,
    IPIV: ndarray<number>,
    B: ndarray<number>,
    LDB: number,
    WORK: ndarray<number>) {

        let func = emlapack.cwrap('dsytrs_aa_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [in] WORK: DOUBLE[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
        ]);

        let LWORK = -1;

        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let aWORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));
        aWORK.set(WORK.data, WORK.offset);

        func(pUPLO, pN, pNRHS, pA, pLDA, pIPIV, pB, pLDB, pWORK, pLWORK);

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pUPLO, pN, pNRHS, pA, pLDA, pIPIV, pB, pLDB, pWORK, pLWORK);
}