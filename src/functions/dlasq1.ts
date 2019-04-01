
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasq1(N: number,
    D: ndarray<number>,
    E: ndarray<number>) {

        let func = emlapack.cwrap('dlasq1_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N]
            'number', // [out] WORK: DOUBLE PRECISION[4*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N));
        aE.set(E.data, E.offset);

        let pWORK = emlapack._malloc(8 * (4*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (4*N));

        func(pN, pD, pE, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlasq1': " + INFO);
        }
}