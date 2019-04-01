
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlassq(N: number,
    X: ndarray<number>,
    INCX: number,
    SCALE: number,
    SUMSQ: number) {

        let func = emlapack.cwrap('dlassq_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] X: DOUBLE PRECISION[N]
            'number', // [in] INCX: INTEGER
            'number', // [in,out] SCALE: DOUBLE PRECISION
            'number', // [in,out] SUMSQ: DOUBLE PRECISION
        ]);

        let pN = emlapack._malloc(4);
        let pINCX = emlapack._malloc(4);
        let pSCALE = emlapack._malloc(8);
        let pSUMSQ = emlapack._malloc(8);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pINCX, INCX, 'i32');
        emlapack.setValue(pSCALE, SCALE, 'double');
        emlapack.setValue(pSUMSQ, SUMSQ, 'double');

        let pX = emlapack._malloc(8 * (N));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (N));
        aX.set(X.data, X.offset);

        func(pN, pX, pINCX, pSCALE, pSUMSQ);
}