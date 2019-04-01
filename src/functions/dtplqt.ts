
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtplqt(M: number,
    N: number,
    L: number,
    MB: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    LDT: number): ndarray<number> {

        let func = emlapack.cwrap('dtplqt_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] L: INTEGER
            'number', // [in] MB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,M]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [out] T: DOUBLE PRECISION[LDT,N]
            'number', // [in] LDT: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MB*M]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pL = emlapack._malloc(4);
        let pMB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pL, L, 'i32');
        emlapack.setValue(pMB, MB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDT, LDT, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (M));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (M));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pT = emlapack._malloc(8 * (LDT) * (N));
        let T = new Float64Array(emlapack.HEAPF64.buffer, pT, (LDT) * (N));

        let pWORK = emlapack._malloc(8 * (MB*M));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (MB*M));

        func(pM, pN, pL, pMB, pA, pLDA, pB, pLDB, pT, pLDT, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtplqt': " + INFO);
        }
        return ndarray(T, [(LDT), (N)]);
}