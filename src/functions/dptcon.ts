
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dptcon(N: number,
    D: ndarray<number>,
    E: ndarray<number>,
    ANORM: number): number {

        let func = emlapack.cwrap('dptcon_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E: DOUBLE PRECISION[N-1]
            'number', // [in] ANORM: DOUBLE PRECISION
            'number', // [out] RCOND: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pANORM = emlapack._malloc(8);
        let pRCOND = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pANORM, ANORM, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pWORK = emlapack._malloc(8 * (N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N));

        func(pN, pD, pE, pANORM, pRCOND, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dptcon': " + INFO);
        }
        return emlapack.getValue(pRCOND, 'double');
}