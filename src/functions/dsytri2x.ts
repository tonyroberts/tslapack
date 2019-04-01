
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsytri2x(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    IPIV: ndarray<number>,
    NB: number) {

        let func = emlapack.cwrap('dsytri2x_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] IPIV: INTEGER[N]
            'number', // [out] WORK: DOUBLE PRECISION[N+NB+1,NB+3]
            'number', // [in] NB: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pNB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pNB, NB, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pWORK = emlapack._malloc(8 * (N+NB+1) * (NB+3));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N+NB+1) * (NB+3));

        func(pUPLO, pN, pA, pLDA, pIPIV, pWORK, pNB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsytri2x': " + INFO);
        }
}