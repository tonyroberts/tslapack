
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgttrs(TRANS: string,
    N: number,
    NRHS: number,
    DL: ndarray<number>,
    D: ndarray<number>,
    DU: ndarray<number>,
    DU2: ndarray<number>,
    IPIV: ndarray<number>,
    B: ndarray<number>,
    LDB: number) {

        let func = emlapack.cwrap('dgttrs_', null, [
            'number', // [in] TRANS: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] DL: DOUBLE PRECISION[N-1]
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] DU: DOUBLE PRECISION[N-1]
            'number', // [in] DU2: DOUBLE PRECISION[N-2]
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pTRANS = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');

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

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        func(pTRANS, pN, pNRHS, pDL, pD, pDU, pDU2, pIPIV, pB, pLDB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgttrs': " + INFO);
        }
}