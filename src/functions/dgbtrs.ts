
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgbtrs(TRANS: string,
    N: number,
    KL: number,
    KU: number,
    NRHS: number,
    AB: ndarray<number>,
    LDAB: number,
    IPIV: ndarray<number>,
    B: ndarray<number>,
    LDB: number) {

        let func = emlapack.cwrap('dgbtrs_', null, [
            'number', // [in] TRANS: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KL: INTEGER
            'number', // [in] KU: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pTRANS = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKL = emlapack._malloc(4);
        let pKU = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKL, KL, 'i32');
        emlapack.setValue(pKU, KU, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        func(pTRANS, pN, pKL, pKU, pNRHS, pAB, pLDAB, pIPIV, pB, pLDB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgbtrs': " + INFO);
        }
}