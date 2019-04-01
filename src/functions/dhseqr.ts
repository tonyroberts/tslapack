
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dhseqr(JOB: string,
    COMPZ: string,
    N: number,
    ILO: number,
    IHI: number,
    H: ndarray<number>,
    LDH: number,
    Z: ndarray<number>,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dhseqr_', null, [
            'number', // [in] JOB: CHARACTER
            'number', // [in] COMPZ: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] ILO: INTEGER
            'number', // [in] IHI: INTEGER
            'number', // [in,out] H: DOUBLE PRECISION[LDH,N]
            'number', // [in] LDH: INTEGER
            'number', // [out] WR: DOUBLE PRECISION[N]
            'number', // [out] WI: DOUBLE PRECISION[N]
            'number', // [in,out] Z: DOUBLE PRECISION[LDZ,N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOB = emlapack._malloc(1);
        let pCOMPZ = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pILO = emlapack._malloc(4);
        let pIHI = emlapack._malloc(4);
        let pLDH = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOB, JOB.charCodeAt(0), 'i8');
        emlapack.setValue(pCOMPZ, COMPZ.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pILO, ILO, 'i32');
        emlapack.setValue(pIHI, IHI, 'i32');
        emlapack.setValue(pLDH, LDH, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pH = emlapack._malloc(8 * (LDH) * (N));
        let aH = new Float64Array(emlapack.HEAPF64.buffer, pH, (LDH) * (N));
        aH.set(H.data, H.offset);

        let pWR = emlapack._malloc(8 * (N));
        let WR = new Float64Array(emlapack.HEAPF64.buffer, pWR, (N));

        let pWI = emlapack._malloc(8 * (N));
        let WI = new Float64Array(emlapack.HEAPF64.buffer, pWI, (N));

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));
        aZ.set(Z.data, Z.offset);

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        func(pJOB, pCOMPZ, pN, pILO, pIHI, pH, pLDH, pWR, pWI, pZ, pLDZ, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dhseqr': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOB, pCOMPZ, pN, pILO, pIHI, pH, pLDH, pWR, pWI, pZ, pLDZ, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dhseqr': " + INFO);
        }

        return ndarray(WI, [(N)]);
}