
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpftrf(TRANSR: string,
    UPLO: string,
    N: number,
    A: ndarray<number>) {

        let func = emlapack.cwrap('dpftrf_', null, [
            'number', // [in] TRANSR: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pTRANSR = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANSR, TRANSR.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');

        let pA = emlapack._malloc(8 * (N*(N+1)/2));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (N*(N+1)/2));
        aA.set(A.data, A.offset);

        func(pTRANSR, pUPLO, pN, pA, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpftrf': " + INFO);
        }
}