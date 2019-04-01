
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasd4(N: number,
    I: number,
    D: ndarray<number>,
    Z: ndarray<number>,
    RHO: number): number {

        let func = emlapack.cwrap('dlasd4_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] I: INTEGER
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] Z: DOUBLE PRECISION[N]
            'number', // [out] DELTA: DOUBLE PRECISION[N]
            'number', // [in] RHO: DOUBLE PRECISION
            'number', // [out] SIGMA: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pI = emlapack._malloc(4);
        let pRHO = emlapack._malloc(8);
        let pSIGMA = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pI, I, 'i32');
        emlapack.setValue(pRHO, RHO, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pZ = emlapack._malloc(8 * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (N));
        aZ.set(Z.data, Z.offset);

        let pDELTA = emlapack._malloc(8 * (N));
        let DELTA = new Float64Array(emlapack.HEAPF64.buffer, pDELTA, (N));

        let pWORK = emlapack._malloc(8 * (N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N));

        func(pN, pI, pD, pZ, pDELTA, pRHO, pSIGMA, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlasd4': " + INFO);
        }
        return emlapack.getValue(pSIGMA, 'double');
}