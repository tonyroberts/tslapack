
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsytri_3x(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    E: ndarray<number>,
    IPIV: ndarray<number>,
    NB: number) {

        let func = emlapack.cwrap('dsytri_3x_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] E: DOUBLE PRECISION[N]
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

        let pE = emlapack._malloc(8 * (N));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N));
        aE.set(E.data, E.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pWORK = emlapack._malloc(8 * (N+NB+1) * (NB+3));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N+NB+1) * (NB+3));

        func(pUPLO, pN, pA, pLDA, pE, pIPIV, pWORK, pNB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsytri_3x': " + INFO);
        }
}