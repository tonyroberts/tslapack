
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dppequ(UPLO: string,
    N: number,
    AP: ndarray<number>): number {

        let func = emlapack.cwrap('dppequ_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] S: DOUBLE PRECISION[N]
            'number', // [out] SCOND: DOUBLE PRECISION
            'number', // [out] AMAX: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pSCOND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pS = emlapack._malloc(8 * (N));
        let S = new Float64Array(emlapack.HEAPF64.buffer, pS, (N));

        func(pUPLO, pN, pAP, pS, pSCOND, pAMAX, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dppequ': " + INFO);
        }
        return emlapack.getValue(pAMAX, 'double');
}