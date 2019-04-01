
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlat2s(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    LDSA: number): ndarray<number> {

        let func = emlapack.cwrap('dlat2s_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] SA: REAL[LDSA,N]
            'number', // [in] LDSA: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDSA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDSA, LDSA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pSA = emlapack._malloc(8 * (LDSA) * (N));
        let SA = new Float64Array(emlapack.HEAPF64.buffer, pSA, (LDSA) * (N));

        func(pUPLO, pN, pA, pLDA, pSA, pLDSA, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlat2s': " + INFO);
        }
        return ndarray(SA, [(LDSA), (N)]);
}