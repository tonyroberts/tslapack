
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaed9(K: number,
    KSTART: number,
    KSTOP: number,
    N: number,
    LDQ: number,
    RHO: number,
    DLAMDA: ndarray<number>,
    W: ndarray<number>,
    LDS: number): ndarray<number> {

        let func = emlapack.cwrap('dlaed9_', null, [
            'number', // [in] K: INTEGER
            'number', // [in] KSTART: INTEGER
            'number', // [in] KSTOP: INTEGER
            'number', // [in] N: INTEGER
            'number', // [out] D: DOUBLE PRECISION[N]
            'number', // [out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [in] RHO: DOUBLE PRECISION
            'number', // [in] DLAMDA: DOUBLE PRECISION[K]
            'number', // [in] W: DOUBLE PRECISION[K]
            'number', // [out] S: DOUBLE PRECISION[LDS, K]
            'number', // [in] LDS: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pK = emlapack._malloc(4);
        let pKSTART = emlapack._malloc(4);
        let pKSTOP = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pRHO = emlapack._malloc(8);
        let pLDS = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pK, K, 'i32');
        emlapack.setValue(pKSTART, KSTART, 'i32');
        emlapack.setValue(pKSTOP, KSTOP, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pRHO, RHO, 'double');
        emlapack.setValue(pLDS, LDS, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let Q = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));

        let pDLAMDA = emlapack._malloc(8 * (K));
        let aDLAMDA = new Float64Array(emlapack.HEAPF64.buffer, pDLAMDA, (K));
        aDLAMDA.set(DLAMDA.data, DLAMDA.offset);

        let pW = emlapack._malloc(8 * (K));
        let aW = new Float64Array(emlapack.HEAPF64.buffer, pW, (K));
        aW.set(W.data, W.offset);

        let pS = emlapack._malloc(8 * (LDS) * (K));
        let S = new Float64Array(emlapack.HEAPF64.buffer, pS, (LDS) * (K));

        func(pK, pKSTART, pKSTOP, pN, pD, pQ, pLDQ, pRHO, pDLAMDA, pW, pS, pLDS, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlaed9': " + INFO);
        }
        return ndarray(S, [(LDS), (K)]);
}