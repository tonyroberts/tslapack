
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarscl2(M: number,
    N: number,
    D: ndarray<number>,
    X: ndarray<number>,
    LDX: number) {

        let func = emlapack.cwrap('dlarscl2_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] D: DOUBLE PRECISION[M]
            'number', // [in,out] X: DOUBLE PRECISION[LDX,N]
            'number', // [in] LDX: INTEGER
        ]);

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDX = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDX, LDX, 'i32');

        let pD = emlapack._malloc(8 * (M));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (M));
        aD.set(D.data, D.offset);

        let pX = emlapack._malloc(8 * (LDX) * (N));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (N));
        aX.set(X.data, X.offset);

        func(pM, pN, pD, pX, pLDX);
}