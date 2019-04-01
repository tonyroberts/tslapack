
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dla_lin_berr(N: number,
    NZ: number,
    NRHS: number,
    RES: ndarray<number>,
    AYB: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dla_lin_berr_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] NZ: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] RES: DOUBLE PRECISION[N,NRHS]
            'number', // [in] AYB: DOUBLE PRECISION[N, NRHS]
            'number', // [out] BERR: DOUBLE PRECISION[NRHS]
        ]);

        let pN = emlapack._malloc(4);
        let pNZ = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNZ, NZ, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');

        let pRES = emlapack._malloc(8 * (N) * (NRHS));
        let aRES = new Float64Array(emlapack.HEAPF64.buffer, pRES, (N) * (NRHS));
        aRES.set(RES.data, RES.offset);

        let pAYB = emlapack._malloc(8 * (N) * (NRHS));
        let aAYB = new Float64Array(emlapack.HEAPF64.buffer, pAYB, (N) * (NRHS));
        aAYB.set(AYB.data, AYB.offset);

        let pBERR = emlapack._malloc(8 * (NRHS));
        let BERR = new Float64Array(emlapack.HEAPF64.buffer, pBERR, (NRHS));

        func(pN, pNZ, pNRHS, pRES, pAYB, pBERR);
        return ndarray(BERR, [(NRHS)]);
}