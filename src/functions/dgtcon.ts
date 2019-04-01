
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgtcon(NORM: string,
    N: number,
    DL: ndarray<number>,
    D: ndarray<number>,
    DU: ndarray<number>,
    DU2: ndarray<number>,
    IPIV: ndarray<number>,
    ANORM: number): number {

        let func = emlapack.cwrap('dgtcon_', null, [
            'number', // [in] NORM: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] DL: DOUBLE PRECISION[N-1]
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] DU: DOUBLE PRECISION[N-1]
            'number', // [in] DU2: DOUBLE PRECISION[N-2]
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in] ANORM: DOUBLE PRECISION
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pNORM = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pANORM = emlapack._malloc(8);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pNORM, NORM.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pANORM, ANORM, 'double');

        let pDL = emlapack._malloc(8 * (N-1));
        let aDL = new Float64Array(emlapack.HEAPF64.buffer, pDL, (N-1));
        aDL.set(DL.data, DL.offset);

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pDU = emlapack._malloc(8 * (N-1));
        let aDU = new Float64Array(emlapack.HEAPF64.buffer, pDU, (N-1));
        aDU.set(DU.data, DU.offset);

        let pDU2 = emlapack._malloc(8 * (N-2));
        let aDU2 = new Float64Array(emlapack.HEAPF64.buffer, pDU2, (N-2));
        aDU2.set(DU2.data, DU2.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pNORM, pN, pDL, pD, pDU, pDU2, pIPIV, pANORM, pRCOND, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgtcon': " + INFO);
        }
        return emlapack.getValue(pRCOND, 'double');
}