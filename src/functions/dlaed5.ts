
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaed5(I: number,
    D: ndarray<number>,
    Z: ndarray<number>,
    RHO: number): number {

        let func = emlapack.cwrap('dlaed5_', null, [
            'number', // [in] I: INTEGER
            'number', // [in] D: DOUBLE PRECISION[2]
            'number', // [in] Z: DOUBLE PRECISION[2]
            'number', // [out] DELTA: DOUBLE PRECISION[2]
            'number', // [in] RHO: DOUBLE PRECISION
            'number', // [out] DLAM: DOUBLE PRECISION
        ]);

        let pI = emlapack._malloc(4);
        let pRHO = emlapack._malloc(8);
        let pDLAM = emlapack._malloc(8);

        emlapack.setValue(pI, I, 'i32');
        emlapack.setValue(pRHO, RHO, 'double');

        let pD = emlapack._malloc(8 * (2));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (2));
        aD.set(D.data, D.offset);

        let pZ = emlapack._malloc(8 * (2));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (2));
        aZ.set(Z.data, Z.offset);

        let pDELTA = emlapack._malloc(8 * (2));
        let DELTA = new Float64Array(emlapack.HEAPF64.buffer, pDELTA, (2));

        func(pI, pD, pZ, pDELTA, pRHO, pDLAM);
        return emlapack.getValue(pDLAM, 'double');
}