
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgbequb(M: number,
    N: number,
    KL: number,
    KU: number,
    AB: ndarray<number>,
    LDAB: number): number {

        let func = emlapack.cwrap('dgbequb_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] KL: INTEGER
            'number', // [in] KU: INTEGER
            'number', // [in] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] R: DOUBLE PRECISION[M]
            'number', // [out] C: DOUBLE PRECISION[N]
            'number', // [out] ROWCND: DOUBLE PRECISION
            'number', // [out] COLCND: DOUBLE PRECISION
            'number', // [out] AMAX: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pKL = emlapack._malloc(4);
        let pKU = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pROWCND = emlapack._malloc(8);
        let pCOLCND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKL, KL, 'i32');
        emlapack.setValue(pKU, KU, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pR = emlapack._malloc(8 * (M));
        let R = new Float64Array(emlapack.HEAPF64.buffer, pR, (M));

        let pC = emlapack._malloc(8 * (N));
        let C = new Float64Array(emlapack.HEAPF64.buffer, pC, (N));

        func(pM, pN, pKL, pKU, pAB, pLDAB, pR, pC, pROWCND, pCOLCND, pAMAX, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgbequb': " + INFO);
        }
        return emlapack.getValue(pAMAX, 'double');
}