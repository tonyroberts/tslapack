
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlartgs(X: number,
    Y: number,
    SIGMA: number): number {

        let func = emlapack.cwrap('dlartgs_', null, [
            'number', // [in] X: DOUBLE PRECISION
            'number', // [in] Y: DOUBLE PRECISION
            'number', // [in] SIGMA: DOUBLE PRECISION
            'number', // [out] CS: DOUBLE PRECISION
            'number', // [out] SN: DOUBLE PRECISION
        ]);

        let pX = emlapack._malloc(8);
        let pY = emlapack._malloc(8);
        let pSIGMA = emlapack._malloc(8);
        let pCS = emlapack._malloc(8);
        let pSN = emlapack._malloc(8);

        emlapack.setValue(pX, X, 'double');
        emlapack.setValue(pY, Y, 'double');
        emlapack.setValue(pSIGMA, SIGMA, 'double');

        func(pX, pY, pSIGMA, pCS, pSN);
        return emlapack.getValue(pSN, 'double');
}