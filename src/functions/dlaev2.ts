
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaev2(A: number,
    B: number,
    C: number): number {

        let func = emlapack.cwrap('dlaev2_', null, [
            'number', // [in] A: DOUBLE PRECISION
            'number', // [in] B: DOUBLE PRECISION
            'number', // [in] C: DOUBLE PRECISION
            'number', // [out] RT1: DOUBLE PRECISION
            'number', // [out] RT2: DOUBLE PRECISION
            'number', // [out] CS1: DOUBLE PRECISION
            'number', // [out] SN1: DOUBLE PRECISION
        ]);

        let pA = emlapack._malloc(8);
        let pB = emlapack._malloc(8);
        let pC = emlapack._malloc(8);
        let pRT1 = emlapack._malloc(8);
        let pRT2 = emlapack._malloc(8);
        let pCS1 = emlapack._malloc(8);
        let pSN1 = emlapack._malloc(8);

        emlapack.setValue(pA, A, 'double');
        emlapack.setValue(pB, B, 'double');
        emlapack.setValue(pC, C, 'double');

        func(pA, pB, pC, pRT1, pRT2, pCS1, pSN1);
        return emlapack.getValue(pSN1, 'double');
}