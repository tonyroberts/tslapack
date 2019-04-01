
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dspev(JOBZ: string,
    UPLO: string,
    N: number,
    AP: ndarray<number>,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dspev_', null, [
            'number', // [in] JOBZ: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[3*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOBZ = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBZ, JOBZ.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let Z = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));

        let pWORK = emlapack._malloc(8 * (3*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (3*N));

        func(pJOBZ, pUPLO, pN, pAP, pW, pZ, pLDZ, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dspev': " + INFO);
        }
        return ndarray(Z, [(LDZ), (N)]);
}