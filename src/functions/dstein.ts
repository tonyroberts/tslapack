
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dstein(N: number,
    D: ndarray<number>,
    E: ndarray<number>,
    M: number,
    W: ndarray<number>,
    IBLOCK: ndarray<number>,
    ISPLIT: ndarray<number>,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dstein_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in] E: DOUBLE PRECISION[N-1]
            'number', // [in] M: INTEGER
            'number', // [in] W: DOUBLE PRECISION[N]
            'number', // [in] IBLOCK: INTEGER[N]
            'number', // [in] ISPLIT: INTEGER[N]
            'number', // [out] Z: DOUBLE PRECISION[LDZ, M]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[5*N]
            'number', // [out] IWORK: INTEGER[N]
            'number', // [out] IFAIL: INTEGER[M]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pW = emlapack._malloc(8 * (N));
        let aW = new Float64Array(emlapack.HEAPF64.buffer, pW, (N));
        aW.set(W.data, W.offset);

        let pIBLOCK = emlapack._malloc(4 * (N));
        let aIBLOCK = new Int32Array(emlapack.HEAPI32.buffer, pIBLOCK, (N));
        aIBLOCK.set(IBLOCK.data, IBLOCK.offset);

        let pISPLIT = emlapack._malloc(4 * (N));
        let aISPLIT = new Int32Array(emlapack.HEAPI32.buffer, pISPLIT, (N));
        aISPLIT.set(ISPLIT.data, ISPLIT.offset);

        let pZ = emlapack._malloc(8 * (LDZ) * (M));
        let Z = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (M));

        let pWORK = emlapack._malloc(8 * (5*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (5*N));

        let pIWORK = emlapack._malloc(4 * (N));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (N));

        let pIFAIL = emlapack._malloc(4 * (M));
        let IFAIL = new Int32Array(emlapack.HEAPI32.buffer, pIFAIL, (M));

        func(pN, pD, pE, pM, pW, pIBLOCK, pISPLIT, pZ, pLDZ, pWORK, pIWORK, pIFAIL, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dstein': " + INFO);
        }
        return ndarray(IFAIL, [(M)]);
}