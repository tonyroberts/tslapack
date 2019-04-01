
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlag2(A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    SAFMIN: number): number {

        let func = emlapack.cwrap('dlag2_', null, [
            'number', // [in] A: DOUBLE PRECISION[LDA, 2]
            'number', // [in] LDA: INTEGER
            'number', // [in] B: DOUBLE PRECISION[LDB, 2]
            'number', // [in] LDB: INTEGER
            'number', // [in] SAFMIN: DOUBLE PRECISION
            'number', // [out] SCALE1: DOUBLE PRECISION
            'number', // [out] SCALE2: DOUBLE PRECISION
            'number', // [out] WR1: DOUBLE PRECISION
            'number', // [out] WR2: DOUBLE PRECISION
            'number', // [out] WI: DOUBLE PRECISION
        ]);

        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pSAFMIN = emlapack._malloc(8);
        let pSCALE1 = emlapack._malloc(8);
        let pSCALE2 = emlapack._malloc(8);
        let pWR1 = emlapack._malloc(8);
        let pWR2 = emlapack._malloc(8);
        let pWI = emlapack._malloc(8);

        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pSAFMIN, SAFMIN, 'double');

        let pA = emlapack._malloc(8 * (LDA) * (2));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (2));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (2));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (2));
        aB.set(B.data, B.offset);

        func(pA, pLDA, pB, pLDB, pSAFMIN, pSCALE1, pSCALE2, pWR1, pWR2, pWI);
        return emlapack.getValue(pWI, 'double');
}