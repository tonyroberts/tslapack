
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function ddisna(JOB: string,
    M: number,
    N: number,
    D: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('ddisna_', null, [
            'number', // [in] JOB: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] D: DOUBLE PRECISION[M]
            'number', // [out] SEP: DOUBLE PRECISION[M]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOB = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOB, JOB.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');

        let pD = emlapack._malloc(8 * (M));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (M));
        aD.set(D.data, D.offset);

        let pSEP = emlapack._malloc(8 * (M));
        let SEP = new Float64Array(emlapack.HEAPF64.buffer, pSEP, (M));

        func(pJOB, pM, pN, pD, pSEP, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'ddisna': " + INFO);
        }
        return ndarray(SEP, [(M)]);
}