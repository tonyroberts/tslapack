
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgebal(JOB: string,
    N: number,
    A: ndarray<number>,
    LDA: number): ndarray<number> {

        let func = emlapack.cwrap('dgebal_', null, [
            'number', // [in] JOB: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] ILO: INTEGER
            'number', // [out] IHI: INTEGER
            'number', // [out] SCALE: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOB = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pILO = emlapack._malloc(4);
        let pIHI = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOB, JOB.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pSCALE = emlapack._malloc(8 * (N));
        let SCALE = new Float64Array(emlapack.HEAPF64.buffer, pSCALE, (N));

        func(pJOB, pN, pA, pLDA, pILO, pIHI, pSCALE, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgebal': " + INFO);
        }
        return ndarray(SCALE, [(N)]);
}