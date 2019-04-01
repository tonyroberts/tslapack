
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpbequ(UPLO: string,
    N: number,
    KD: number,
    AB: ndarray<number>,
    LDAB: number): number {

        let func = emlapack.cwrap('dpbequ_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] S: DOUBLE PRECISION[N]
            'number', // [out] SCOND: DOUBLE PRECISION
            'number', // [out] AMAX: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKD = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pSCOND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pS = emlapack._malloc(8 * (N));
        let S = new Float64Array(emlapack.HEAPF64.buffer, pS, (N));

        func(pUPLO, pN, pKD, pAB, pLDAB, pS, pSCOND, pAMAX, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpbequ': " + INFO);
        }
        return emlapack.getValue(pAMAX, 'double');
}