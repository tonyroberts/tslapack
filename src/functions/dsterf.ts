
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsterf(N: number,
    D: ndarray<number>,
    E: ndarray<number>) {

        let func = emlapack.cwrap('dsterf_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N-1]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        func(pN, pD, pE, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsterf': " + INFO);
        }
}