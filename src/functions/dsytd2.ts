
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsytd2(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dsytd2_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] D: DOUBLE PRECISION[N]
            'number', // [out] E: DOUBLE PRECISION[N-1]
            'number', // [out] TAU: DOUBLE PRECISION[N-1]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pD = emlapack._malloc(8 * (N));
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));

        let pE = emlapack._malloc(8 * (N-1));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));

        let pTAU = emlapack._malloc(8 * (N-1));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (N-1));

        func(pUPLO, pN, pA, pLDA, pD, pE, pTAU, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsytd2': " + INFO);
        }
        return ndarray(TAU, [(N-1)]);
}