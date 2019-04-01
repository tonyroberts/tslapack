
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarnv(IDIST: number,
    ISEED: ndarray<number>,
    N: number): ndarray<number> {

        let func = emlapack.cwrap('dlarnv_', null, [
            'number', // [in] IDIST: INTEGER
            'number', // [in,out] ISEED: INTEGER[4]
            'number', // [in] N: INTEGER
            'number', // [out] X: DOUBLE PRECISION[N]
        ]);

        let pIDIST = emlapack._malloc(4);
        let pN = emlapack._malloc(4);

        emlapack.setValue(pIDIST, IDIST, 'i32');
        emlapack.setValue(pN, N, 'i32');

        let pISEED = emlapack._malloc(4 * (4));
        let aISEED = new Int32Array(emlapack.HEAPI32.buffer, pISEED, (4));
        aISEED.set(ISEED.data, ISEED.offset);

        let pX = emlapack._malloc(8 * (N));
        let X = new Float64Array(emlapack.HEAPF64.buffer, pX, (N));

        func(pIDIST, pISEED, pN, pX);
        return ndarray(X, [(N)]);
}