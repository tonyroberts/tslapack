
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaqsb(UPLO: string,
    N: number,
    KD: number,
    AB: ndarray<number>,
    LDAB: number,
    S: ndarray<number>,
    SCOND: number,
    AMAX: number): string {

        let func = emlapack.cwrap('dlaqsb_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [in] S: DOUBLE PRECISION[N]
            'number', // [in] SCOND: DOUBLE PRECISION
            'number', // [in] AMAX: DOUBLE PRECISION
            'number', // [out] EQUED: CHARACTER
        ]);

        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKD = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pSCOND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pEQUED = emlapack._malloc(1);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pSCOND, SCOND, 'double');
        emlapack.setValue(pAMAX, AMAX, 'double');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pS = emlapack._malloc(8 * (N));
        let aS = new Float64Array(emlapack.HEAPF64.buffer, pS, (N));
        aS.set(S.data, S.offset);

        func(pUPLO, pN, pKD, pAB, pLDAB, pS, pSCOND, pAMAX, pEQUED);
        return emlapack.getValue(pEQUED, 'i8');
}