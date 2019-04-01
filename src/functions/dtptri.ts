
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtptri(UPLO: string,
    DIAG: string,
    N: number,
    AP: ndarray<number>) {

        let func = emlapack.cwrap('dtptri_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] DIAG: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pDIAG = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pDIAG, DIAG.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        func(pUPLO, pDIAG, pN, pAP, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtptri': " + INFO);
        }
}