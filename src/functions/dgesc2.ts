
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgesc2(N: number,
    A: ndarray<number>,
    LDA: number,
    RHS: ndarray<number>,
    IPIV: ndarray<number>,
    JPIV: ndarray<number>): number {

        let func = emlapack.cwrap('dgesc2_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] RHS: DOUBLE PRECISION[N]
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in] JPIV: INTEGER[N]
            'number', // [out] SCALE: DOUBLE PRECISION
        ]);

        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pSCALE = emlapack._malloc(8);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pRHS = emlapack._malloc(8 * (N));
        let aRHS = new Float64Array(emlapack.HEAPF64.buffer, pRHS, (N));
        aRHS.set(RHS.data, RHS.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pJPIV = emlapack._malloc(4 * (N));
        let aJPIV = new Int32Array(emlapack.HEAPI32.buffer, pJPIV, (N));
        aJPIV.set(JPIV.data, JPIV.offset);

        func(pN, pA, pLDA, pRHS, pIPIV, pJPIV, pSCALE);
        return emlapack.getValue(pSCALE, 'double');
}