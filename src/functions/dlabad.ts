
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlabad(SMALL: number,
    LARGE: number) {

        let func = emlapack.cwrap('dlabad_', null, [
            'number', // [in,out] SMALL: DOUBLE PRECISION
            'number', // [in,out] LARGE: DOUBLE PRECISION
        ]);

        let pSMALL = emlapack._malloc(8);
        let pLARGE = emlapack._malloc(8);

        emlapack.setValue(pSMALL, SMALL, 'double');
        emlapack.setValue(pLARGE, LARGE, 'double');

        func(pSMALL, pLARGE);
}