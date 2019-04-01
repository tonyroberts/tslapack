
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlas2(F: number,
    G: number,
    H: number): number {

        let func = emlapack.cwrap('dlas2_', null, [
            'number', // [in] F: DOUBLE PRECISION
            'number', // [in] G: DOUBLE PRECISION
            'number', // [in] H: DOUBLE PRECISION
            'number', // [out] SSMIN: DOUBLE PRECISION
            'number', // [out] SSMAX: DOUBLE PRECISION
        ]);

        let pF = emlapack._malloc(8);
        let pG = emlapack._malloc(8);
        let pH = emlapack._malloc(8);
        let pSSMIN = emlapack._malloc(8);
        let pSSMAX = emlapack._malloc(8);

        emlapack.setValue(pF, F, 'double');
        emlapack.setValue(pG, G, 'double');
        emlapack.setValue(pH, H, 'double');

        func(pF, pG, pH, pSSMIN, pSSMAX);
        return emlapack.getValue(pSSMAX, 'double');
}