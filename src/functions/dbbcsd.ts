
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dbbcsd(JOBU1: string,
    JOBU2: string,
    JOBV1T: string,
    JOBV2T: string,
    TRANS: string,
    M: number,
    P: number,
    Q: number,
    THETA: ndarray<number>,
    PHI: ndarray<number>,
    U1: ndarray<number>,
    LDU1: number,
    U2: ndarray<number>,
    LDU2: number,
    V1T: ndarray<number>,
    LDV1T: number,
    V2T: ndarray<number>,
    LDV2T: number): ndarray<number> {

        let func = emlapack.cwrap('dbbcsd_', null, [
            'number', // [in] JOBU1: CHARACTER
            'number', // [in] JOBU2: CHARACTER
            'number', // [in] JOBV1T: CHARACTER
            'number', // [in] JOBV2T: CHARACTER
            'number', // [in] TRANS: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] P: INTEGER
            'number', // [in] Q: INTEGER
            'number', // [in,out] THETA: DOUBLE PRECISION[Q]
            'number', // [in,out] PHI: DOUBLE PRECISION[Q-1]
            'number', // [in,out] U1: DOUBLE PRECISION[LDU1,P]
            'number', // [in] LDU1: INTEGER
            'number', // [in,out] U2: DOUBLE PRECISION[LDU2,M-P]
            'number', // [in] LDU2: INTEGER
            'number', // [in,out] V1T: DOUBLE PRECISION[LDV1T,Q]
            'number', // [in] LDV1T: INTEGER
            'number', // [in,out] V2T: DOUBLE PRECISION[LDV2T,M-Q]
            'number', // [in] LDV2T: INTEGER
            'number', // [out] B11D: DOUBLE PRECISION[Q]
            'number', // [out] B11E: DOUBLE PRECISION[Q-1]
            'number', // [out] B12D: DOUBLE PRECISION[Q]
            'number', // [out] B12E: DOUBLE PRECISION[Q-1]
            'number', // [out] B21D: DOUBLE PRECISION[Q]
            'number', // [out] B21E: DOUBLE PRECISION[Q-1]
            'number', // [out] B22D: DOUBLE PRECISION[Q]
            'number', // [out] B22E: DOUBLE PRECISION[Q-1]
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOBU1 = emlapack._malloc(1);
        let pJOBU2 = emlapack._malloc(1);
        let pJOBV1T = emlapack._malloc(1);
        let pJOBV2T = emlapack._malloc(1);
        let pTRANS = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pP = emlapack._malloc(4);
        let pQ = emlapack._malloc(4);
        let pLDU1 = emlapack._malloc(4);
        let pLDU2 = emlapack._malloc(4);
        let pLDV1T = emlapack._malloc(4);
        let pLDV2T = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBU1, JOBU1.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBU2, JOBU2.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBV1T, JOBV1T.charCodeAt(0), 'i8');
        emlapack.setValue(pJOBV2T, JOBV2T.charCodeAt(0), 'i8');
        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pP, P, 'i32');
        emlapack.setValue(pQ, Q, 'i32');
        emlapack.setValue(pLDU1, LDU1, 'i32');
        emlapack.setValue(pLDU2, LDU2, 'i32');
        emlapack.setValue(pLDV1T, LDV1T, 'i32');
        emlapack.setValue(pLDV2T, LDV2T, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pTHETA = emlapack._malloc(8 * (Q));
        let aTHETA = new Float64Array(emlapack.HEAPF64.buffer, pTHETA, (Q));
        aTHETA.set(THETA.data, THETA.offset);

        let pPHI = emlapack._malloc(8 * (Q-1));
        let aPHI = new Float64Array(emlapack.HEAPF64.buffer, pPHI, (Q-1));
        aPHI.set(PHI.data, PHI.offset);

        let pU1 = emlapack._malloc(8 * (LDU1) * (P));
        let aU1 = new Float64Array(emlapack.HEAPF64.buffer, pU1, (LDU1) * (P));
        aU1.set(U1.data, U1.offset);

        let pU2 = emlapack._malloc(8 * (LDU2) * (M-P));
        let aU2 = new Float64Array(emlapack.HEAPF64.buffer, pU2, (LDU2) * (M-P));
        aU2.set(U2.data, U2.offset);

        let pV1T = emlapack._malloc(8 * (LDV1T) * (Q));
        let aV1T = new Float64Array(emlapack.HEAPF64.buffer, pV1T, (LDV1T) * (Q));
        aV1T.set(V1T.data, V1T.offset);

        let pV2T = emlapack._malloc(8 * (LDV2T) * (M-Q));
        let aV2T = new Float64Array(emlapack.HEAPF64.buffer, pV2T, (LDV2T) * (M-Q));
        aV2T.set(V2T.data, V2T.offset);

        let pB11D = emlapack._malloc(8 * (Q));
        let B11D = new Float64Array(emlapack.HEAPF64.buffer, pB11D, (Q));

        let pB11E = emlapack._malloc(8 * (Q-1));
        let B11E = new Float64Array(emlapack.HEAPF64.buffer, pB11E, (Q-1));

        let pB12D = emlapack._malloc(8 * (Q));
        let B12D = new Float64Array(emlapack.HEAPF64.buffer, pB12D, (Q));

        let pB12E = emlapack._malloc(8 * (Q-1));
        let B12E = new Float64Array(emlapack.HEAPF64.buffer, pB12E, (Q-1));

        let pB21D = emlapack._malloc(8 * (Q));
        let B21D = new Float64Array(emlapack.HEAPF64.buffer, pB21D, (Q));

        let pB21E = emlapack._malloc(8 * (Q-1));
        let B21E = new Float64Array(emlapack.HEAPF64.buffer, pB21E, (Q-1));

        let pB22D = emlapack._malloc(8 * (Q));
        let B22D = new Float64Array(emlapack.HEAPF64.buffer, pB22D, (Q));

        let pB22E = emlapack._malloc(8 * (Q-1));
        let B22E = new Float64Array(emlapack.HEAPF64.buffer, pB22E, (Q-1));

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pJOBU1, pJOBU2, pJOBV1T, pJOBV2T, pTRANS, pM, pP, pQ, pTHETA, pPHI, pU1, pLDU1, pU2, pLDU2, pV1T, pLDV1T, pV2T, pLDV2T, pB11D, pB11E, pB12D, pB12E, pB21D, pB21E, pB22D, pB22E, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dbbcsd': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOBU1, pJOBU2, pJOBV1T, pJOBV2T, pTRANS, pM, pP, pQ, pTHETA, pPHI, pU1, pLDU1, pU2, pLDU2, pV1T, pLDV1T, pV2T, pLDV2T, pB11D, pB11E, pB12D, pB12E, pB21D, pB21E, pB22D, pB22E, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dbbcsd': " + INFO);
        }

        return ndarray(B22E, [(Q-1)]);
}