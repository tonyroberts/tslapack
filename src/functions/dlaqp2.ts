
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaqp2(M: number,
    N: number,
    OFFSET: number,
    A: ndarray<number>,
    LDA: number,
    JPVT: ndarray<number>,
    VN1: ndarray<number>,
    VN2: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dlaqp2_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] OFFSET: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] JPVT: INTEGER[N]
            'number', // [out] TAU: DOUBLE PRECISION[min(M,N)]
            'number', // [in,out] VN1: DOUBLE PRECISION[N]
            'number', // [in,out] VN2: DOUBLE PRECISION[N]
            'number', // [out] WORK: DOUBLE PRECISION[N]
        ]);

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pOFFSET = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pOFFSET, OFFSET, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pJPVT = emlapack._malloc(4 * (N));
        let aJPVT = new Int32Array(emlapack.HEAPI32.buffer, pJPVT, (N));
        aJPVT.set(JPVT.data, JPVT.offset);

        let pTAU = emlapack._malloc(8 * (Math.min(M,N)));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (Math.min(M,N)));

        let pVN1 = emlapack._malloc(8 * (N));
        let aVN1 = new Float64Array(emlapack.HEAPF64.buffer, pVN1, (N));
        aVN1.set(VN1.data, VN1.offset);

        let pVN2 = emlapack._malloc(8 * (N));
        let aVN2 = new Float64Array(emlapack.HEAPF64.buffer, pVN2, (N));
        aVN2.set(VN2.data, VN2.offset);

        let pWORK = emlapack._malloc(8 * (N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N));

        func(pM, pN, pOFFSET, pA, pLDA, pJPVT, pTAU, pVN1, pVN2, pWORK);
        return ndarray(TAU, [(Math.min(M,N))]);
}