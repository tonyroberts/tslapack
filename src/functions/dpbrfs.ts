
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpbrfs(UPLO: string,
    N: number,
    KD: number,
    NRHS: number,
    AB: ndarray<number>,
    LDAB: number,
    AFB: ndarray<number>,
    LDAFB: number,
    B: ndarray<number>,
    LDB: number,
    X: ndarray<number>,
    LDX: number): ndarray<number> {

        let func = emlapack.cwrap('dpbrfs_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [in] AFB: DOUBLE PRECISION[LDAFB,N]
            'number', // [in] LDAFB: INTEGER
            'number', // [in] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [in,out] X: DOUBLE PRECISION[LDX,NRHS]
            'number', // [in] LDX: INTEGER
            'number', // [out] FERR: DOUBLE PRECISION[NRHS]
            'number', // [out] BERR: DOUBLE PRECISION[NRHS]
            'number', // [out] WORK: DOUBLE PRECISION[3*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKD = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLDAFB = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDX = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLDAFB, LDAFB, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDX, LDX, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pAFB = emlapack._malloc(8 * (LDAFB) * (N));
        let aAFB = new Float64Array(emlapack.HEAPF64.buffer, pAFB, (LDAFB) * (N));
        aAFB.set(AFB.data, AFB.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        let pX = emlapack._malloc(8 * (LDX) * (NRHS));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (NRHS));
        aX.set(X.data, X.offset);

        let pFERR = emlapack._malloc(8 * (NRHS));
        let FERR = new Float64Array(emlapack.HEAPF64.buffer, pFERR, (NRHS));

        let pBERR = emlapack._malloc(8 * (NRHS));
        let BERR = new Float64Array(emlapack.HEAPF64.buffer, pBERR, (NRHS));

        let pWORK = emlapack._malloc(8 * (3*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (3*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        func(pUPLO, pN, pKD, pNRHS, pAB, pLDAB, pAFB, pLDAFB, pB, pLDB, pX, pLDX, pFERR, pBERR, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpbrfs': " + INFO);
        }
        return ndarray(BERR, [(NRHS)]);
}