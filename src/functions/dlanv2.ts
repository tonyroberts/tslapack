
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlanv2(A: number,
    B: number,
    C: number,
    D: number): number {

        let func = emlapack.cwrap('dlanv2_', null, [
            'number', // [in,out] A: DOUBLE PRECISION
            'number', // [in,out] B: DOUBLE PRECISION
            'number', // [in,out] C: DOUBLE PRECISION
            'number', // [in,out] D: DOUBLE PRECISION
            'number', // [out] RT1R: DOUBLE PRECISION
            'number', // [out] RT1I: DOUBLE PRECISION
            'number', // [out] RT2R: DOUBLE PRECISION
            'number', // [out] RT2I: DOUBLE PRECISION
            'number', // [out] CS: DOUBLE PRECISION
            'number', // [out] SN: DOUBLE PRECISION
        ]);

        let pA = emlapack._malloc(8);
        let pB = emlapack._malloc(8);
        let pC = emlapack._malloc(8);
        let pD = emlapack._malloc(8);
        let pRT1R = emlapack._malloc(8);
        let pRT1I = emlapack._malloc(8);
        let pRT2R = emlapack._malloc(8);
        let pRT2I = emlapack._malloc(8);
        let pCS = emlapack._malloc(8);
        let pSN = emlapack._malloc(8);

        emlapack.setValue(pA, A, 'double');
        emlapack.setValue(pB, B, 'double');
        emlapack.setValue(pC, C, 'double');
        emlapack.setValue(pD, D, 'double');

        func(pA, pB, pC, pD, pRT1R, pRT1I, pRT2R, pRT2I, pCS, pSN);
        return emlapack.getValue(pSN, 'double');
}