
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaic1(JOB: number,
    J: number,
    X: ndarray<number>,
    SEST: number,
    W: ndarray<number>,
    GAMMA: number): number {

        let func = emlapack.cwrap('dlaic1_', null, [
            'number', // [in] JOB: INTEGER
            'number', // [in] J: INTEGER
            'number', // [in] X: DOUBLE PRECISION[J]
            'number', // [in] SEST: DOUBLE PRECISION
            'number', // [in] W: DOUBLE PRECISION[J]
            'number', // [in] GAMMA: DOUBLE PRECISION
            'number', // [out] SESTPR: DOUBLE PRECISION
            'number', // [out] S: DOUBLE PRECISION
            'number', // [out] C: DOUBLE PRECISION
        ]);

        let pJOB = emlapack._malloc(4);
        let pJ = emlapack._malloc(4);
        let pSEST = emlapack._malloc(8);
        let pGAMMA = emlapack._malloc(8);
        let pSESTPR = emlapack._malloc(8);
        let pS = emlapack._malloc(8);
        let pC = emlapack._malloc(8);

        emlapack.setValue(pJOB, JOB, 'i32');
        emlapack.setValue(pJ, J, 'i32');
        emlapack.setValue(pSEST, SEST, 'double');
        emlapack.setValue(pGAMMA, GAMMA, 'double');

        let pX = emlapack._malloc(8 * (J));
        let aX = new Float64Array(emlapack.HEAPF64.buffer, pX, (J));
        aX.set(X.data, X.offset);

        let pW = emlapack._malloc(8 * (J));
        let aW = new Float64Array(emlapack.HEAPF64.buffer, pW, (J));
        aW.set(W.data, W.offset);

        func(pJOB, pJ, pX, pSEST, pW, pGAMMA, pSESTPR, pS, pC);
        return emlapack.getValue(pC, 'double');
}