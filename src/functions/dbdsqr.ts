
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dbdsqr(UPLO: string,
    N: number,
    NCVT: number,
    NRU: number,
    NCC: number,
    D: ndarray<number>,
    E: ndarray<number>,
    VT: ndarray<number>,
    LDVT: number,
    U: ndarray<number>,
    LDU: number,
    C: ndarray<number>,
    LDC: number) {

        let func = emlapack.cwrap('dbdsqr_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NCVT: INTEGER
            'number', // [in] NRU: INTEGER
            'number', // [in] NCC: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N-1]
            'number', // [in,out] VT: DOUBLE PRECISION[LDVT, NCVT]
            'number', // [in] LDVT: INTEGER
            'number', // [in,out] U: DOUBLE PRECISION[LDU, N]
            'number', // [in] LDU: INTEGER
            'number', // [in,out] C: DOUBLE PRECISION[LDC, NCC]
            'number', // [in] LDC: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[4*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNCVT = emlapack._malloc(4);
        let pNRU = emlapack._malloc(4);
        let pNCC = emlapack._malloc(4);
        let pLDVT = emlapack._malloc(4);
        let pLDU = emlapack._malloc(4);
        let pLDC = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNCVT, NCVT, 'i32');
        emlapack.setValue(pNRU, NRU, 'i32');
        emlapack.setValue(pNCC, NCC, 'i32');
        emlapack.setValue(pLDVT, LDVT, 'i32');
        emlapack.setValue(pLDU, LDU, 'i32');
        emlapack.setValue(pLDC, LDC, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pVT = emlapack._malloc(8 * (LDVT) * (NCVT));
        let aVT = new Float64Array(emlapack.HEAPF64.buffer, pVT, (LDVT) * (NCVT));
        aVT.set(VT.data, VT.offset);

        let pU = emlapack._malloc(8 * (LDU) * (N));
        let aU = new Float64Array(emlapack.HEAPF64.buffer, pU, (LDU) * (N));
        aU.set(U.data, U.offset);

        let pC = emlapack._malloc(8 * (LDC) * (NCC));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (LDC) * (NCC));
        aC.set(C.data, C.offset);

        let pWORK = emlapack._malloc(8 * (4*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (4*N));

        func(pUPLO, pN, pNCVT, pNRU, pNCC, pD, pE, pVT, pLDVT, pU, pLDU, pC, pLDC, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dbdsqr': " + INFO);
        }
}