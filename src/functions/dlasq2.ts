
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasq2(N: number,
    Z: ndarray<number>) {

        let func = emlapack.cwrap('dlasq2_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] Z: DOUBLE PRECISION[4*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');

        let pZ = emlapack._malloc(8 * (4*N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (4*N));
        aZ.set(Z.data, Z.offset);

        func(pN, pZ, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlasq2': " + INFO);
        }
}