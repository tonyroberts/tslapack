
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsytrd_sy2sb(UPLO: string,
    N: number,
    KD: number,
    A: ndarray<number>,
    LDA: number,
    LDAB: number): ndarray<number> {

        let func = emlapack.cwrap('dsytrd_sy2sb_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] TAU: DOUBLE PRECISION[N-KD]
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKD = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let AB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));

        let pTAU = emlapack._malloc(8 * (N-KD));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (N-KD));

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        func(pUPLO, pN, pKD, pA, pLDA, pAB, pLDAB, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsytrd_sy2sb': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pUPLO, pN, pKD, pA, pLDA, pAB, pLDAB, pTAU, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsytrd_sy2sb': " + INFO);
        }

        return ndarray(TAU, [(N-KD)]);
}