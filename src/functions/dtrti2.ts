
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtrti2(UPLO: string,
    DIAG: string,
    N: number,
    A: ndarray<number>,
    LDA: number) {

        let func = emlapack.cwrap('dtrti2_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] DIAG: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pDIAG = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pDIAG, DIAG.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        func(pUPLO, pDIAG, pN, pA, pLDA, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtrti2': " + INFO);
        }
}