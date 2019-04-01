
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlagv2(A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number): number {

        let func = emlapack.cwrap('dlagv2_', null, [
            'number', // [in,out] A: DOUBLE PRECISION[LDA, 2]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB, 2]
            'number', // [in] LDB: INTEGER
            'number', // [out] ALPHAR: DOUBLE PRECISION[2]
            'number', // [out] ALPHAI: DOUBLE PRECISION[2]
            'number', // [out] BETA: DOUBLE PRECISION[2]
            'number', // [out] CSL: DOUBLE PRECISION
            'number', // [out] SNL: DOUBLE PRECISION
            'number', // [out] CSR: DOUBLE PRECISION
            'number', // [out] SNR: DOUBLE PRECISION
        ]);

        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pCSL = emlapack._malloc(8);
        let pSNL = emlapack._malloc(8);
        let pCSR = emlapack._malloc(8);
        let pSNR = emlapack._malloc(8);

        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (2));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (2));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (2));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (2));
        aB.set(B.data, B.offset);

        let pALPHAR = emlapack._malloc(8 * (2));
        let ALPHAR = new Float64Array(emlapack.HEAPF64.buffer, pALPHAR, (2));

        let pALPHAI = emlapack._malloc(8 * (2));
        let ALPHAI = new Float64Array(emlapack.HEAPF64.buffer, pALPHAI, (2));

        let pBETA = emlapack._malloc(8 * (2));
        let BETA = new Float64Array(emlapack.HEAPF64.buffer, pBETA, (2));

        func(pA, pLDA, pB, pLDB, pALPHAR, pALPHAI, pBETA, pCSL, pSNL, pCSR, pSNR);
        return emlapack.getValue(pSNR, 'double');
}