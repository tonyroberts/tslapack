
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarrc(JOBT: string,
    N: number,
    VL: number,
    VU: number,
    D: ndarray<number>,
    E: ndarray<number>,
    PIVMIN: number): number {

        let func = emlapack.cwrap('dlarrc_', null, [
            'number', // [in] JOBT: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] VL: DOUBLE PRECISION
            'number', // [in] VU: DOUBLE PRECISION
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E: DOUBLE PRECISION[N]
            'number', // [in] PIVMIN: DOUBLE PRECISION
            'number', // [out] EIGCNT: INTEGER
            'number', // [out] LCNT: INTEGER
            'number', // [out] RCNT: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOBT = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pVL = emlapack._malloc(8);
        let pVU = emlapack._malloc(8);
        let pPIVMIN = emlapack._malloc(8);
        let pEIGCNT = emlapack._malloc(4);
        let pLCNT = emlapack._malloc(4);
        let pRCNT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBT, JOBT.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pVL, VL, 'double');
        emlapack.setValue(pVU, VU, 'double');
        emlapack.setValue(pPIVMIN, PIVMIN, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N));
        aE.set(E.data, E.offset);

        func(pJOBT, pN, pVL, pVU, pD, pE, pPIVMIN, pEIGCNT, pLCNT, pRCNT, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarrc': " + INFO);
        }
        return emlapack.getValue(pRCNT, 'i32');
}