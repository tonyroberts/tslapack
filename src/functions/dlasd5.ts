
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasd5(I: number,
    D: ndarray<number>,
    Z: ndarray<number>,
    RHO: number): number {

        let func = emlapack.cwrap('dlasd5_', null, [
            'number', // [in] I: INTEGER
            'number', // [in] D: DOUBLE PRECISION[2]
            'number', // [in] Z: DOUBLE PRECISION[2]
            'number', // [out] DELTA: DOUBLE PRECISION[2]
            'number', // [in] RHO: DOUBLE PRECISION
            'number', // [out] DSIGMA: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[2]
        ]);

        let pI = emlapack._malloc(4);
        let pRHO = emlapack._malloc(8);
        let pDSIGMA = emlapack._malloc(8);

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

        let pWORK = emlapack._malloc(8 * (2));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2));

        func(pI, pD, pZ, pDELTA, pRHO, pDSIGMA, pWORK);
        return emlapack.getValue(pDSIGMA, 'double');
}