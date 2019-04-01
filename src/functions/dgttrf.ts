
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgttrf(N: number,
    DL: ndarray<number>,
    D: ndarray<number>,
    DU: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dgttrf_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] DL: DOUBLE PRECISION[N-1]
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] DU: DOUBLE PRECISION[N-1]
            'number', // [out] DU2: DOUBLE PRECISION[N-2]
            'number', // [out] IPIV: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');

        let pDL = emlapack._malloc(8 * (N-1));
        let aDL = new Float64Array(emlapack.HEAPF64.buffer, pDL, (N-1));
        aDL.set(DL.data, DL.offset);

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pDU = emlapack._malloc(8 * (N-1));
        let aDU = new Float64Array(emlapack.HEAPF64.buffer, pDU, (N-1));
        aDU.set(DU.data, DU.offset);

        let pDU2 = emlapack._malloc(8 * (N-2));
        let DU2 = new Float64Array(emlapack.HEAPF64.buffer, pDU2, (N-2));

        let pIPIV = emlapack._malloc(4 * (N));
        let IPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));

        func(pN, pDL, pD, pDU, pDU2, pIPIV, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgttrf': " + INFO);
        }
        return ndarray(IPIV, [(N)]);
}