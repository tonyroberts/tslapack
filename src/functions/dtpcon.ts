
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtpcon(NORM: string,
    UPLO: string,
    DIAG: string,
    N: number,
    AP: ndarray<number>): number {

        let func = emlapack.cwrap('dtpcon_', null, [
            'number', // [in] NORM: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] DIAG: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[3*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pNORM = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pDIAG = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pNORM, NORM.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pDIAG, DIAG.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pWORK = emlapack._malloc(8 * (3*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (3*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pNORM, pUPLO, pDIAG, pN, pAP, pRCOND, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtpcon': " + INFO);
        }
        return emlapack.getValue(pRCOND, 'double');
}