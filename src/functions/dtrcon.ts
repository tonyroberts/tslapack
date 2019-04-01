
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtrcon(NORM: string,
    UPLO: string,
    DIAG: string,
    N: number,
    A: ndarray<number>,
    LDA: number): number {

        let func = emlapack.cwrap('dtrcon_', null, [
            'number', // [in] NORM: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] DIAG: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[3*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pNORM = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pDIAG = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pNORM, NORM.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pDIAG, DIAG.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pWORK = emlapack._malloc(8 * (3*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (3*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pNORM, pUPLO, pDIAG, pN, pA, pLDA, pRCOND, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtrcon': " + INFO);
        }
        return emlapack.getValue(pRCOND, 'double');
}