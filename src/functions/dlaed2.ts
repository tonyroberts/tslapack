
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaed2(N: number,
    N1: number,
    D: ndarray<number>,
    Q: ndarray<number>,
    LDQ: number,
    INDXQ: ndarray<number>,
    RHO: number,
    Z: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dlaed2_', null, [
            'number', // [out] K: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] N1: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ, N]
            'number', // [in] LDQ: INTEGER
            'number', // [in,out] INDXQ: INTEGER[N]
            'number', // [in,out] RHO: DOUBLE PRECISION
            'number', // [in] Z: DOUBLE PRECISION[N]
            'number', // [out] DLAMDA: DOUBLE PRECISION[N]
            'number', // [out] W: DOUBLE PRECISION[N]
            'number', // [out] Q2: DOUBLE PRECISION[N1**2+(N-N1)**2]
            'number', // [out] INDX: INTEGER[N]
            'number', // [out] INDXC: INTEGER[N]
            'number', // [out] INDXP: INTEGER[N]
            'number', // [out] COLTYP: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pK = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pN1 = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pRHO = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pN1, N1, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pRHO, RHO, 'double');

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

        let pW = emlapack._malloc(8 * (N));
        let W = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));

        let pQ2 = emlapack._malloc(8 * (N1**2+(N-N1)**2));
        let Q2 = new Float64Array(emlapack.HEAPF64.buffer, pQ2, (N1**2+(N-N1)**2));

        let pINDX = emlapack._malloc(4 * (N));
        let INDX = new Int32Array(emlapack.HEAPI32.buffer, pINDX, (N));

        let pINDXC = emlapack._malloc(4 * (N));
        let INDXC = new Int32Array(emlapack.HEAPI32.buffer, pINDXC, (N));

        let pINDXP = emlapack._malloc(4 * (N));
        let INDXP = new Int32Array(emlapack.HEAPI32.buffer, pINDXP, (N));

        let pCOLTYP = emlapack._malloc(4 * (N));
        let COLTYP = new Int32Array(emlapack.HEAPI32.buffer, pCOLTYP, (N));

        func(pK, pN, pN1, pD, pQ, pLDQ, pINDXQ, pRHO, pZ, pDLAMDA, pW, pQ2, pINDX, pINDXC, pINDXP, pCOLTYP, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlaed2': " + INFO);
        }
        return ndarray(COLTYP, [(N)]);
}