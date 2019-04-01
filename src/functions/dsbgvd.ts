
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsbgvd(JOBZ: string,
    UPLO: string,
    N: number,
    KA: number,
    KB: number,
    AB: ndarray<number>,
    LDAB: number,
    BB: ndarray<number>,
    LDBB: number,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dsbgvd_', null, [
            'number', // [in] JOBZ: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KA: INTEGER
            'number', // [in] KB: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB, N]
            'number', // [in] LDAB: INTEGER
            'number', // [in,out] BB: DOUBLE PRECISION[LDBB, N]
            'number', // [in] LDBB: INTEGER
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] IWORK: INTEGER[MAX(1,LIWORK)]
            'number', // [in] LIWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;
        let LIWORK = -1;

        let pJOBZ = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKA = emlapack._malloc(4);
        let pKB = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLDBB = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pLIWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBZ, JOBZ.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKA, KA, 'i32');
        emlapack.setValue(pKB, KB, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLDBB, LDBB, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        emlapack.setValue(pLIWORK, LIWORK, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pBB = emlapack._malloc(8 * (LDBB) * (N));
        let aBB = new Float64Array(emlapack.HEAPF64.buffer, pBB, (LDBB) * (N));
        aBB.set(BB.data, BB.offset);

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let Z = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        let pIWORK = emlapack._malloc(4 * (Math.max(1,LIWORK)));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (Math.max(1,LIWORK)));

        func(pJOBZ, pUPLO, pN, pKA, pKB, pAB, pLDAB, pBB, pLDBB, pW, pZ, pLDZ, pWORK, pLWORK, pIWORK, pLIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsbgvd': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);
        LIWORK = emlapack.getValue(pIWORK, 'i32');
        emlapack.setValue(pLIWORK, LIWORK, 'i32');
        pIWORK = emlapack._malloc(4 * LIWORK);

        func(pJOBZ, pUPLO, pN, pKA, pKB, pAB, pLDAB, pBB, pLDBB, pW, pZ, pLDZ, pWORK, pLWORK, pIWORK, pLIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsbgvd': " + INFO);
        }

        return ndarray(Z, [(LDZ), (N)]);
}