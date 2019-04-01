
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dorbdb(TRANS: string,
    SIGNS: string,
    M: number,
    P: number,
    Q: number,
    X11: ndarray<number>,
    LDX11: number,
    X12: ndarray<number>,
    LDX12: number,
    X21: ndarray<number>,
    LDX21: number,
    X22: ndarray<number>,
    LDX22: number): ndarray<number> {

        let func = emlapack.cwrap('dorbdb_', null, [
            'number', // [in] TRANS: CHARACTER
            'number', // [in] SIGNS: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in] Q: INTEGER
            'number', // [in,out] X11: DOUBLE PRECISION[LDX11,Q]
            'number', // [in] LDX11: INTEGER
            'number', // [in,out] X12: DOUBLE PRECISION[LDX12,M-Q]
            'number', // [in] LDX12: INTEGER
            'number', // [in,out] X21: DOUBLE PRECISION[LDX21,Q]
            'number', // [in] LDX21: INTEGER
            'number', // [in,out] X22: DOUBLE PRECISION[LDX22,M-Q]
            'number', // [in] LDX22: INTEGER
            'number', // [out] THETA: DOUBLE PRECISION[Q]
            'number', // [out] PHI: DOUBLE PRECISION[Q-1]
            'number', // [out] TAUP1: DOUBLE PRECISION[P]
            'number', // [out] TAUP2: DOUBLE PRECISION[M-P]
            'number', // [out] TAUQ1: DOUBLE PRECISION[Q]
            'number', // [out] TAUQ2: DOUBLE PRECISION[M-Q]
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pTRANS = emlapack._malloc(1);
        let pSIGNS = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pQ = emlapack._malloc(4);
        let pLDX11 = emlapack._malloc(4);
        let pLDX12 = emlapack._malloc(4);
        let pLDX21 = emlapack._malloc(4);
        let pLDX22 = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pSIGNS, SIGNS.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pQ, Q, 'i32');
        emlapack.setValue(pLDX11, LDX11, 'i32');
        emlapack.setValue(pLDX12, LDX12, 'i32');
        emlapack.setValue(pLDX21, LDX21, 'i32');
        emlapack.setValue(pLDX22, LDX22, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pX11 = emlapack._malloc(8 * (LDX11) * (Q));
        let aX11 = new Float64Array(emlapack.HEAPF64.buffer, pX11, (LDX11) * (Q));
        aX11.set(X11.data, X11.offset);

        let pX12 = emlapack._malloc(8 * (LDX12) * (M-Q));
        let aX12 = new Float64Array(emlapack.HEAPF64.buffer, pX12, (LDX12) * (M-Q));
        aX12.set(X12.data, X12.offset);

        let pX21 = emlapack._malloc(8 * (LDX21) * (Q));
        let aX21 = new Float64Array(emlapack.HEAPF64.buffer, pX21, (LDX21) * (Q));
        aX21.set(X21.data, X21.offset);

        let pX22 = emlapack._malloc(8 * (LDX22) * (M-Q));
        let aX22 = new Float64Array(emlapack.HEAPF64.buffer, pX22, (LDX22) * (M-Q));
        aX22.set(X22.data, X22.offset);

        let pTHETA = emlapack._malloc(8 * (Q));
        let THETA = new Float64Array(emlapack.HEAPF64.buffer, pTHETA, (Q));

        let pPHI = emlapack._malloc(8 * (Q-1));
        let PHI = new Float64Array(emlapack.HEAPF64.buffer, pPHI, (Q-1));

        let pTAUP1 = emlapack._malloc(8 * (P));
        let TAUP1 = new Float64Array(emlapack.HEAPF64.buffer, pTAUP1, (P));

        let pTAUP2 = emlapack._malloc(8 * (M-P));
        let TAUP2 = new Float64Array(emlapack.HEAPF64.buffer, pTAUP2, (M-P));

        let pTAUQ1 = emlapack._malloc(8 * (Q));
        let TAUQ1 = new Float64Array(emlapack.HEAPF64.buffer, pTAUQ1, (Q));

        let pTAUQ2 = emlapack._malloc(8 * (M-Q));
        let TAUQ2 = new Float64Array(emlapack.HEAPF64.buffer, pTAUQ2, (M-Q));

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        func(pTRANS, pSIGNS, pM, pP, pQ, pX11, pLDX11, pX12, pLDX12, pX21, pLDX21, pX22, pLDX22, pTHETA, pPHI, pTAUP1, pTAUP2, pTAUQ1, pTAUQ2, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorbdb': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pTRANS, pSIGNS, pM, pP, pQ, pX11, pLDX11, pX12, pLDX12, pX21, pLDX21, pX22, pLDX22, pTHETA, pPHI, pTAUP1, pTAUP2, pTAUQ1, pTAUQ2, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorbdb': " + INFO);
        }

        return ndarray(TAUQ2, [(M-Q)]);
}