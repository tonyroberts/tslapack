
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsbev(JOBZ: string,
    UPLO: string,
    N: number,
    KD: number,
    AB: ndarray<number>,
    LDAB: number,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dsbev_', null, [
            'number', // [in] JOBZ: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KD: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB, N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[max(1,3*N-2)]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOBZ = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKD = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBZ, JOBZ.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKD, KD, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let Z = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,3*N-2)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,3*N-2)));

        func(pJOBZ, pUPLO, pN, pKD, pAB, pLDAB, pW, pZ, pLDZ, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsbev': " + INFO);
        }
        return ndarray(Z, [(LDZ), (N)]);
}