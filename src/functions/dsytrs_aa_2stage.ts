
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsytrs_aa_2stage(UPLO: string,
    N: number,
    NRHS: number,
    A: ndarray<number>,
    LDA: number,
    LTB: number,
    IPIV: ndarray<number>,
    IPIV2: ndarray<number>,
    B: ndarray<number>,
    LDB: number): ndarray<number> {

        let func = emlapack.cwrap('dsytrs_aa_2stage_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] TB: DOUBLE PRECISION[LTB]
            'number', // [in] LTB: INTEGER
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in] IPIV2: INTEGER[N]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLTB = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLTB, LTB, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pTB = emlapack._malloc(8 * (LTB));
        let TB = new Float64Array(emlapack.HEAPF64.buffer, pTB, (LTB));

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pIPIV2 = emlapack._malloc(4 * (N));
        let aIPIV2 = new Int32Array(emlapack.HEAPI32.buffer, pIPIV2, (N));
        aIPIV2.set(IPIV2.data, IPIV2.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        func(pUPLO, pN, pNRHS, pA, pLDA, pTB, pLTB, pIPIV, pIPIV2, pB, pLDB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsytrs_aa_2stage': " + INFO);
        }
        return ndarray(TB, [(LTB)]);
}