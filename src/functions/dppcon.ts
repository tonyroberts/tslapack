
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dppcon(UPLO: string,
    N: number,
    AP: ndarray<number>,
    ANORM: number): number {

        let func = emlapack.cwrap('dppcon_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [in] ANORM: DOUBLE PRECISION
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[3*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pANORM = emlapack._malloc(8);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pANORM, ANORM, 'double');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pWORK = emlapack._malloc(8 * (3*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (3*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pUPLO, pN, pAP, pANORM, pRCOND, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dppcon': " + INFO);
        }
        return emlapack.getValue(pRCOND, 'double');
}