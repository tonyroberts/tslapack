
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlagts(JOB: number,
    N: number,
    A: ndarray<number>,
    B: ndarray<number>,
    C: ndarray<number>,
    D: ndarray<number>,
    IN: ndarray<number>,
    Y: ndarray<number>,
    TOL: number) {

        let func = emlapack.cwrap('dlagts_', null, [
            'number', // [in] JOB: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[N]
            'number', // [in] B: DOUBLE PRECISION[N-1]
            'number', // [in] C: DOUBLE PRECISION[N-1]
            'number', // [in] D: DOUBLE PRECISION[N-2]
            'number', // [in] IN: INTEGER[N]
            'number', // [in,out] Y: DOUBLE PRECISION[N]
            'number', // [in,out] TOL: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOB = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pTOL = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOB, JOB, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pTOL, TOL, 'double');

        let pA = emlapack._malloc(8 * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (N-1));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (N-1));
        aB.set(B.data, B.offset);

        let pC = emlapack._malloc(8 * (N-1));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (N-1));
        aC.set(C.data, C.offset);

        let pD = emlapack._malloc(8 * (N-2));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N-2));
        aD.set(D.data, D.offset);

        let pIN = emlapack._malloc(4 * (N));
        let aIN = new Int32Array(emlapack.HEAPI32.buffer, pIN, (N));
        aIN.set(IN.data, IN.offset);

        let pY = emlapack._malloc(8 * (N));
        let aY = new Float64Array(emlapack.HEAPF64.buffer, pY, (N));
        aY.set(Y.data, Y.offset);

        func(pJOB, pN, pA, pB, pC, pD, pIN, pY, pTOL, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlagts': " + INFO);
        }
}