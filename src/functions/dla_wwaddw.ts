
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dla_wwaddw(N: number,
    X: ndarray<number>,
    Y: ndarray<number>,
    W: ndarray<number>) {

        let func = emlapack.cwrap('dla_wwaddw_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] X: DOUBLE PRECISION[N]
            'number', // [in,out] Y: DOUBLE PRECISION[N]
            'number', // [in] W: DOUBLE PRECISION[N]
        ]);

        let pN = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');

        let pX = emlapack._malloc(8 * (N));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (N));
        aX.set(X.data, X.offset);

        let pY = emlapack._malloc(8 * (N));
        let aY = new Float64Array(emlapack.HEAPF64.buffer, pY, (N));
        aY.set(Y.data, Y.offset);

        let pW = emlapack._malloc(8 * (N));
        let aW = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));
        aW.set(W.data, W.offset);

        func(pN, pX, pY, pW);
}