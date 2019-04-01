
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtpttf(TRANSR: string,
    UPLO: string,
    N: number,
    AP: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dtpttf_', null, [
            'number', // [in] TRANSR: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] ARF: DOUBLE PRECISION[N*(N+1)/2]
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

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pARF = emlapack._malloc(8 * (N*(N+1)/2));
        let ARF = new Float64Array(emlapack.HEAPF64.buffer, pARF, (N*(N+1)/2));

        func(pTRANSR, pUPLO, pN, pAP, pARF, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtpttf': " + INFO);
        }
        return ndarray(ARF, [(N*(N+1)/2)]);
}