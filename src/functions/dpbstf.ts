
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpbstf(UPLO: string,
    N: number,
    KD: number,
    AB: ndarray<number>,
    LDAB: number) {

        let func = emlapack.cwrap('dpbstf_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKD = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        func(pUPLO, pN, pKD, pAB, pLDAB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpbstf': " + INFO);
        }
}