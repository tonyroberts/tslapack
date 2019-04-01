
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsygs2(ITYPE: number,
    UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number) {

        let func = emlapack.cwrap('dsygs2_', null, [
            'number', // [in] ITYPE: INTEGER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pITYPE = emlapack._malloc(4);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pITYPE, ITYPE, 'i32');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        func(pITYPE, pUPLO, pN, pA, pLDA, pB, pLDB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsygs2': " + INFO);
        }
}