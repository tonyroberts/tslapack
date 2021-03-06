
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dorbdb4(M: number,
    P: number,
    Q: number,
    X11: ndarray<number>,
    LDX11: number,
    X21: ndarray<number>,
    LDX21: number): ndarray<number> {

        let func = emlapack.cwrap('dorbdb4_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in] Q: INTEGER
            'number', // [in,out] X11: DOUBLE PRECISION[LDX11,Q]
            'number', // [in] LDX11: INTEGER
            'number', // [in,out] X21: DOUBLE PRECISION[LDX21,Q]
            'number', // [in] LDX21: INTEGER
            'number', // [out] THETA: DOUBLE PRECISION[Q]
            'number', // [out] PHI: DOUBLE PRECISION[Q-1]
            'number', // [out] TAUP1: DOUBLE PRECISION[P]
            'number', // [out] TAUP2: DOUBLE PRECISION[M-P]
            'number', // [out] TAUQ1: DOUBLE PRECISION[Q]
            'number', // [out] PHANTOM: DOUBLE PRECISION[M]
            'number', // [out] WORK: DOUBLE PRECISION[LWORK]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pM = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pQ = emlapack._malloc(4);
        let pLDX11 = emlapack._malloc(4);
        let pLDX21 = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pQ, Q, 'i32');
        emlapack.setValue(pLDX11, LDX11, 'i32');
        emlapack.setValue(pLDX21, LDX21, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pX11 = emlapack._malloc(8 * (LDX11) * (Q));
        let aX11 = new Float64Array(emlapack.HEAPF64.buffer, pX11, (LDX11) * (Q));
        aX11.set(X11.data, X11.offset);

        let pX21 = emlapack._malloc(8 * (LDX21) * (Q));
        let aX21 = new Float64Array(emlapack.HEAPF64.buffer, pX21, (LDX21) * (Q));
        aX21.set(X21.data, X21.offset);

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

        let pPHANTOM = emlapack._malloc(8 * (M));
        let PHANTOM = new Float64Array(emlapack.HEAPF64.buffer, pPHANTOM, (M));

        let pWORK = emlapack._malloc(8 * (LWORK));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (LWORK));

        func(pM, pP, pQ, pX11, pLDX11, pX21, pLDX21, pTHETA, pPHI, pTAUP1, pTAUP2, pTAUQ1, pPHANTOM, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorbdb4': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pM, pP, pQ, pX11, pLDX11, pX21, pLDX21, pTHETA, pPHI, pTAUP1, pTAUP2, pTAUQ1, pPHANTOM, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dorbdb4': " + INFO);
        }

        return ndarray(PHANTOM, [(M)]);
}