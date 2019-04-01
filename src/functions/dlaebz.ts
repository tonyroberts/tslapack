
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaebz(IJOB: number,
    NITMAX: number,
    N: number,
    MMAX: number,
    MINP: number,
    NBMIN: number,
    ABSTOL: number,
    RELTOL: number,
    PIVMIN: number,
    D: ndarray<number>,
    E: ndarray<number>,
    E2: ndarray<number>,
    NVAL: ndarray<number>,
    AB: ndarray<number>,
    C: ndarray<number>,
    NAB: ndarray<number>): number {

        let func = emlapack.cwrap('dlaebz_', null, [
            'number', // [in] IJOB: INTEGER
            'number', // [in] NITMAX: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] MMAX: INTEGER
            'number', // [in] MINP: INTEGER
            'number', // [in] NBMIN: INTEGER
            'number', // [in] ABSTOL: DOUBLE PRECISION
            'number', // [in] RELTOL: DOUBLE PRECISION
            'number', // [in] PIVMIN: DOUBLE PRECISION
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E: DOUBLE PRECISION[N]
            'number', // [in] E2: DOUBLE PRECISION[N]
            'number', // [in,out] NVAL: INTEGER[MINP]
            'number', // [in,out] AB: DOUBLE PRECISION[MMAX,2]
            'number', // [in,out] C: DOUBLE PRECISION[MMAX]
            'number', // [out] MOUT: INTEGER
            'number', // [in,out] NAB: INTEGER[MMAX,2]
            'number', // [out] WORK: DOUBLE PRECISION[MMAX]
            'number', // [out] IWORK: INTEGER[MMAX]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pIJOB = emlapack._malloc(4);
        let pNITMAX = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pMMAX = emlapack._malloc(4);
        let pMINP = emlapack._malloc(4);
        let pNBMIN = emlapack._malloc(4);
        let pABSTOL = emlapack._malloc(8);
        let pRELTOL = emlapack._malloc(8);
        let pPIVMIN = emlapack._malloc(8);
        let pMOUT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pIJOB, IJOB, 'i32');
        emlapack.setValue(pNITMAX, NITMAX, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pMMAX, MMAX, 'i32');
        emlapack.setValue(pMINP, MINP, 'i32');
        emlapack.setValue(pNBMIN, NBMIN, 'i32');
        emlapack.setValue(pABSTOL, ABSTOL, 'double');
        emlapack.setValue(pRELTOL, RELTOL, 'double');
        emlapack.setValue(pPIVMIN, PIVMIN, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N));
        aE.set(E.data, E.offset);

        let pE2 = emlapack._malloc(8 * (N));
        let aE2 = new Float64Array(emlapack.HEAPF64.buffer, pE2, (N));
        aE2.set(E2.data, E2.offset);

        let pNVAL = emlapack._malloc(4 * (MINP));
        let aNVAL = new Int32Array(emlapack.HEAPI32.buffer, pNVAL, (MINP));
        aNVAL.set(NVAL.data, NVAL.offset);

        let pAB = emlapack._malloc(8 * (MMAX) * (2));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (MMAX) * (2));
        aAB.set(AB.data, AB.offset);

        let pC = emlapack._malloc(8 * (MMAX));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (MMAX));
        aC.set(C.data, C.offset);

        let pNAB = emlapack._malloc(4 * (MMAX) * (2));
        let aNAB = new Int32Array(emlapack.HEAPI32.buffer, pNAB, (MMAX) * (2));
        aNAB.set(NAB.data, NAB.offset);

        let pWORK = emlapack._malloc(8 * (MMAX));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (MMAX));

        let pIWORK = emlapack._malloc(4 * (MMAX));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (MMAX));

        func(pIJOB, pNITMAX, pN, pMMAX, pMINP, pNBMIN, pABSTOL, pRELTOL, pPIVMIN, pD, pE, pE2, pNVAL, pAB, pC, pMOUT, pNAB, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlaebz': " + INFO);
        }
        return emlapack.getValue(pMOUT, 'i32');
}