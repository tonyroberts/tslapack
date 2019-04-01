
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasrt(ID: string,
    N: number,
    D: ndarray<number>) {

        let func = emlapack.cwrap('dlasrt_', null, [
            'number', // [in] ID: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pID = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pID, ID.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        func(pID, pN, pD, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlasrt': " + INFO);
        }
}