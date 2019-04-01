
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpstf2(UPLO: string,
    N: number,
    A: ndarray<number>,
    TOL: number,
    LDA: number): number {

        let func = emlapack.cwrap('dpstf2_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [out] PIV: INTEGER[N]
            'number', // [out] RANK: INTEGER
            'number', // [in] TOL: DOUBLE PRECISION
            'number', // [in] LDA: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pRANK = emlapack._malloc(4);
        let pTOL = emlapack._malloc(8);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pTOL, TOL, 'double');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pPIV = emlapack._malloc(4 * (N));
        let PIV = new Int32Array(emlapack.HEAPI32.buffer, pPIV, (N));

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        func(pUPLO, pN, pA, pPIV, pRANK, pTOL, pLDA, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpstf2': " + INFO);
        }
        return emlapack.getValue(pRANK, 'i32');
}