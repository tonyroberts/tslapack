
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlartgp(F: number,
    G: number): number {

        let func = emlapack.cwrap('dlartgp_', null, [
            'number', // [in] F: DOUBLE PRECISION
            'number', // [in] G: DOUBLE PRECISION
            'number', // [out] CS: DOUBLE PRECISION
            'number', // [out] SN: DOUBLE PRECISION
            'number', // [out] R: DOUBLE PRECISION
        ]);

        let pF = emlapack._malloc(8);
        let pG = emlapack._malloc(8);
        let pCS = emlapack._malloc(8);
        let pSN = emlapack._malloc(8);
        let pR = emlapack._malloc(8);

        emlapack.setValue(pF, F, 'double');
        emlapack.setValue(pG, G, 'double');

        func(pF, pG, pCS, pSN, pR);
        return emlapack.getValue(pR, 'double');
}