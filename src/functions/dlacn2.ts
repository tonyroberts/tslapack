
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlacn2(N: number,
    X: ndarray<number>,
    EST: number,
    KASE: number,
    ISAVE: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dlacn2_', null, [
            'number', // [in] N: INTEGER
            'number', // [out] V: DOUBLE PRECISION[N]
            'number', // [in,out] X: DOUBLE PRECISION[N]
            'number', // [out] ISGN: INTEGER[N]
            'number', // [in,out] EST: DOUBLE PRECISION
            'number', // [in,out] KASE: INTEGER
            'number', // [in,out] ISAVE: INTEGER[3]
        ]);

        let pN = emlapack._malloc(4);
        let pEST = emlapack._malloc(8);
        let pKASE = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pEST, EST, 'double');
        emlapack.setValue(pKASE, KASE, 'i32');

        let pV = emlapack._malloc(8 * (N));
        let V = new Float64Array(emlapack.HEAPF64.buffer, pV, (N));

        let pX = emlapack._malloc(8 * (N));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (N));
        aX.set(X.data, X.offset);

        let pISGN = emlapack._malloc(4 * (N));
        let ISGN = new Int32Array(emlapack.HEAPI32.buffer, pISGN, (N));

        let pISAVE = emlapack._malloc(4 * (3));
        let aISAVE = new Int32Array(emlapack.HEAPI32.buffer, pISAVE, (3));
        aISAVE.set(ISAVE.data, ISAVE.offset);

        func(pN, pV, pX, pISGN, pEST, pKASE, pISAVE);
        return ndarray(ISGN, [(N)]);
}