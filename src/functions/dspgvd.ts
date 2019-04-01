
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dspgvd(ITYPE: number,
    JOBZ: string,
    UPLO: string,
    N: number,
    AP: ndarray<number>,
    BP: ndarray<number>,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dspgvd_', null, [
            'number', // [in] ITYPE: INTEGER
            'number', // [in] JOBZ: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [in,out] BP: DOUBLE PRECISION[N*(N+1)/2]
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

        let pITYPE = emlapack._malloc(4);
        let pJOBZ = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pLIWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pITYPE, ITYPE, 'i32');
        emlapack.setValue(pJOBZ, JOBZ.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        emlapack.setValue(pLIWORK, LIWORK, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pBP = emlapack._malloc(8 * (N*(N+1)/2));
        let aBP = new Float64Array(emlapack.HEAPF64.buffer, pBP, (N*(N+1)/2));
        aBP.set(BP.data, BP.offset);

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let Z = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        let pIWORK = emlapack._malloc(4 * (Math.max(1,LIWORK)));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (Math.max(1,LIWORK)));

        func(pITYPE, pJOBZ, pUPLO, pN, pAP, pBP, pW, pZ, pLDZ, pWORK, pLWORK, pIWORK, pLIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dspgvd': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);
        LIWORK = emlapack.getValue(pIWORK, 'i32');
        emlapack.setValue(pLIWORK, LIWORK, 'i32');
        pIWORK = emlapack._malloc(4 * LIWORK);

        func(pITYPE, pJOBZ, pUPLO, pN, pAP, pBP, pW, pZ, pLDZ, pWORK, pLWORK, pIWORK, pLIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dspgvd': " + INFO);
        }

        return ndarray(Z, [(LDZ), (N)]);
}