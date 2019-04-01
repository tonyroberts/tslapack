
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaqr1(N: number,
    H: ndarray<number>,
    LDH: number,
    SR1: number,
    SI1: number,
    SR2: number,
    SI2: number): ndarray<number> {

        let func = emlapack.cwrap('dlaqr1_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] H: DOUBLE PRECISION[LDH,N]
            'number', // [in] LDH: INTEGER
            'number', // [in] SR1: DOUBLE PRECISION
            'number', // [in] SI1: DOUBLE PRECISION
            'number', // [in] SR2: DOUBLE PRECISION
            'number', // [in] SI2: DOUBLE PRECISION
            'number', // [out] V: DOUBLE PRECISION[N]
        ]);

        let pN = emlapack._malloc(4);
        let pLDH = emlapack._malloc(4);
        let pSR1 = emlapack._malloc(8);
        let pSI1 = emlapack._malloc(8);
        let pSR2 = emlapack._malloc(8);
        let pSI2 = emlapack._malloc(8);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDH, LDH, 'i32');
        emlapack.setValue(pSR1, SR1, 'double');
        emlapack.setValue(pSI1, SI1, 'double');
        emlapack.setValue(pSR2, SR2, 'double');
        emlapack.setValue(pSI2, SI2, 'double');

        let pH = emlapack._malloc(8 * (LDH) * (N));
        let aH = new Float64Array(emlapack.HEAPF64.buffer, pH, (LDH) * (N));
        aH.set(H.data, H.offset);

        let pV = emlapack._malloc(8 * (N));
        let V = new Float64Array(emlapack.HEAPF64.buffer, pV, (N));

        func(pN, pH, pLDH, pSR1, pSI1, pSR2, pSI2, pV);
        return ndarray(V, [(N)]);
}