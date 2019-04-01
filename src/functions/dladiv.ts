
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dladiv(A: number,
    B: number,
    C: number,
    D: number): number {

        let func = emlapack.cwrap('dladiv_', null, [
            'number', // [in] A: DOUBLE PRECISION
            'number', // [in] B: DOUBLE PRECISION
            'number', // [in] C: DOUBLE PRECISION
            'number', // [in] D: DOUBLE PRECISION
            'number', // [out] P: DOUBLE PRECISION
            'number', // [out] Q: DOUBLE PRECISION
        ]);

        let pA = emlapack._malloc(8);
        let pB = emlapack._malloc(8);
        let pC = emlapack._malloc(8);
        let pD = emlapack._malloc(8);
        let pP = emlapack._malloc(8);
        let pQ = emlapack._malloc(8);

        emlapack.setValue(pA, A, 'double');
        emlapack.setValue(pB, B, 'double');
        emlapack.setValue(pC, C, 'double');
        emlapack.setValue(pD, D, 'double');

        func(pA, pB, pC, pD, pP, pQ);
        return emlapack.getValue(pQ, 'double');
}