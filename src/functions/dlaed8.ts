
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaed8(ICOMPQ: number,
    N: number,
    QSIZ: number,
    D: ndarray<number>,
    Q: ndarray<number>,
    LDQ: number,
    INDXQ: ndarray<number>,
    RHO: number,
    CUTPNT: number,
    Z: ndarray<number>,
    LDQ2: number): ndarray<number> {

        let func = emlapack.cwrap('dlaed8_', null, [
            'number', // [in] ICOMPQ: INTEGER
            'number', // [out] K: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] QSIZ: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [in] INDXQ: INTEGER[N]
            'number', // [in,out] RHO: DOUBLE PRECISION
            'number', // [in] CUTPNT: INTEGER
            'number', // [in] Z: DOUBLE PRECISION[N]
            'number', // [out] DLAMDA: DOUBLE PRECISION[N]
            'number', // [out] Q2: DOUBLE PRECISION[LDQ2,N]
            'number', // [in] LDQ2: INTEGER
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] PERM: INTEGER[N]
            'number', // [out] GIVPTR: INTEGER
            'number', // [out] GIVCOL: INTEGER[2, N]
            'number', // [out] GIVNUM: DOUBLE PRECISION[2, N]
            'number', // [out] INDXP: INTEGER[N]
            'number', // [out] INDX: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pICOMPQ = emlapack._malloc(4);
        let pK = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pQSIZ = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pRHO = emlapack._malloc(8);
        let pCUTPNT = emlapack._malloc(4);
        let pLDQ2 = emlapack._malloc(4);
        let pGIVPTR = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pICOMPQ, ICOMPQ, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pQSIZ, QSIZ, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pRHO, RHO, 'double');
        emlapack.setValue(pCUTPNT, CUTPNT, 'i32');
        emlapack.setValue(pLDQ2, LDQ2, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let aQ = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));
        aQ.set(Q.data, Q.offset);

        let pINDXQ = emlapack._malloc(4 * (N));
        let aINDXQ = new Int32Array(emlapack.HEAPI32.buffer, pINDXQ, (N));
        aINDXQ.set(INDXQ.data, INDXQ.offset);

        let pZ = emlapack._malloc(8 * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (N));
        aZ.set(Z.data, Z.offset);

        let pDLAMDA = emlapack._malloc(8 * (N));
        let DLAMDA = new Float64Array(emlapack.HEAPF64.buffer, pDLAMDA, (N));

        let pQ2 = emlapack._malloc(8 * (LDQ2) * (N));
        let Q2 = new Float64Array(emlapack.HEAPF64.buffer, pQ2, (LDQ2) * (N));

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pPERM = emlapack._malloc(4 * (N));
        let PERM = new Int32Array(emlapack.HEAPI32.buffer, pPERM, (N));

        let pGIVCOL = emlapack._malloc(4 * (2) * (N));
        let GIVCOL = new Int32Array(emlapack.HEAPI32.buffer, pGIVCOL, (2) * (N));

        let pGIVNUM = emlapack._malloc(8 * (2) * (N));
        let GIVNUM = new Float64Array(emlapack.HEAPF64.buffer, pGIVNUM, (2) * (N));

        let pINDXP = emlapack._malloc(4 * (N));
        let INDXP = new Int32Array(emlapack.HEAPI32.buffer, pINDXP, (N));

        let pINDX = emlapack._malloc(4 * (N));
        let INDX = new Int32Array(emlapack.HEAPI32.buffer, pINDX, (N));

        func(pICOMPQ, pK, pN, pQSIZ, pD, pQ, pLDQ, pINDXQ, pRHO, pCUTPNT, pZ, pDLAMDA, pQ2, pLDQ2, pW, pPERM, pGIVPTR, pGIVCOL, pGIVNUM, pINDXP, pINDX, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlaed8': " + INFO);
        }
        return ndarray(INDX, [(N)]);
}