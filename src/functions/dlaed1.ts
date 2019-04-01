
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaed1(N: number,
    D: ndarray<number>,
    Q: ndarray<number>,
    LDQ: number,
    INDXQ: ndarray<number>,
    RHO: number,
    CUTPNT: number) {

        let func = emlapack.cwrap('dlaed1_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [in,out] INDXQ: INTEGER[N]
            'number', // [in] RHO: DOUBLE PRECISION
            'number', // [in] CUTPNT: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[4*N + N**2]
            'number', // [out] IWORK: INTEGER[4*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pRHO = emlapack._malloc(8);
        let pCUTPNT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pRHO, RHO, 'double');
        emlapack.setValue(pCUTPNT, CUTPNT, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let aQ = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));
        aQ.set(Q.data, Q.offset);

        let pINDXQ = emlapack._malloc(4 * (N));
        let aINDXQ = new Int32Array(emlapack.HEAPI32.buffer, pINDXQ, (N));
        aINDXQ.set(INDXQ.data, INDXQ.offset);

        let pWORK = emlapack._malloc(8 * (4*N+N**2));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (4*N+N**2));

        let pIWORK = emlapack._malloc(4 * (4*N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (4*N));

        func(pN, pD, pQ, pLDQ, pINDXQ, pRHO, pCUTPNT, pWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlaed1': " + INFO);
        }
}