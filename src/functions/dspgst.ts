
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dspgst(ITYPE: number,
    UPLO: string,
    N: number,
    AP: ndarray<number>,
    BP: ndarray<number>) {

        let func = emlapack.cwrap('dspgst_', null, [
            'number', // [in] ITYPE: INTEGER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [in] BP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pITYPE = emlapack._malloc(4);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pITYPE, ITYPE, 'i32');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pBP = emlapack._malloc(8 * (N*(N+1)/2));
        let aBP = new Float64Array(emlapack.HEAPF64.buffer, pBP, (N*(N+1)/2));
        aBP.set(BP.data, BP.offset);

        func(pITYPE, pUPLO, pN, pAP, pBP, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dspgst': " + INFO);
        }
}