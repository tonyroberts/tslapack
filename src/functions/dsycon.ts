
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsycon(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    IPIV: ndarray<number>,
    ANORM: number): number {

        let func = emlapack.cwrap('dsycon_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in] ANORM: DOUBLE PRECISION
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pANORM = emlapack._malloc(8);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pANORM, ANORM, 'double');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pUPLO, pN, pA, pLDA, pIPIV, pANORM, pRCOND, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsycon': " + INFO);
        }
        return emlapack.getValue(pRCOND, 'double');
}