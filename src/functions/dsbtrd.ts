
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsbtrd(VECT: string,
    UPLO: string,
    N: number,
    KD: number,
    AB: ndarray<number>,
    LDAB: number,
    Q: ndarray<number>,
    LDQ: number): ndarray<number> {

        let func = emlapack.cwrap('dsbtrd_', null, [
            'number', // [in] VECT: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] D: DOUBLE PRECISION[N]
            'number', // [out] E: DOUBLE PRECISION[N-1]
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pVECT = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKD = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pVECT, VECT.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pD = emlapack._malloc(8 * (N));
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));

        let pE = emlapack._malloc(8 * (N-1));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let aQ = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));
        aQ.set(Q.data, Q.offset);

        let pWORK = emlapack._malloc(8 * (N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N));

        func(pVECT, pUPLO, pN, pKD, pAB, pLDAB, pD, pE, pQ, pLDQ, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsbtrd': " + INFO);
        }
        return ndarray(E, [(N-1)]);
}