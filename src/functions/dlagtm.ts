
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlagtm(TRANS: string,
    N: number,
    NRHS: number,
    ALPHA: number,
    DL: ndarray<number>,
    D: ndarray<number>,
    DU: ndarray<number>,
    X: ndarray<number>,
    LDX: number,
    BETA: number,
    B: ndarray<number>,
    LDB: number) {

        let func = emlapack.cwrap('dlagtm_', null, [
            'number', // [in] TRANS: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] ALPHA: DOUBLE PRECISION
            'number', // [in] DL: DOUBLE PRECISION[N-1]
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] DU: DOUBLE PRECISION[N-1]
            'number', // [in] X: DOUBLE PRECISION[LDX,NRHS]
            'number', // [in] LDX: INTEGER
            'number', // [in] BETA: DOUBLE PRECISION
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
        ]);

        let pTRANS = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pALPHA = emlapack._malloc(8);
        let pLDX = emlapack._malloc(4);
        let pBETA = emlapack._malloc(8);
        let pLDB = emlapack._malloc(4);

        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pALPHA, ALPHA, 'double');
        emlapack.setValue(pLDX, LDX, 'i32');
        emlapack.setValue(pBETA, BETA, 'double');
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

        let pX = emlapack._malloc(8 * (LDX) * (NRHS));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (NRHS));
        aX.set(X.data, X.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        func(pTRANS, pN, pNRHS, pALPHA, pDL, pD, pDU, pX, pLDX, pBETA, pB, pLDB);
}