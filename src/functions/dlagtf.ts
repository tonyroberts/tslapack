
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlagtf(N: number,
    A: ndarray<number>,
    LAMBDA: number,
    B: ndarray<number>,
    C: ndarray<number>,
    TOL: number): ndarray<number> {

        let func = emlapack.cwrap('dlagtf_', null, [
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[N]
            'number', // [in] LAMBDA: DOUBLE PRECISION
            'number', // [in,out] B: DOUBLE PRECISION[N-1]
            'number', // [in,out] C: DOUBLE PRECISION[N-1]
            'number', // [in] TOL: DOUBLE PRECISION
            'number', // [out] D: DOUBLE PRECISION[N-2]
            'number', // [out] IN: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pLAMBDA = emlapack._malloc(8);
        let pTOL = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLAMBDA, LAMBDA, 'double');
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
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (N-2));

        let pIN = emlapack._malloc(4 * (N));
        let IN = new Int32Array(emlapack.HEAPI32.buffer, pIN, (N));

        func(pN, pA, pLAMBDA, pB, pC, pTOL, pD, pIN, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlagtf': " + INFO);
        }
        return ndarray(IN, [(N)]);
}