
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaqge(M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    R: ndarray<number>,
    C: ndarray<number>,
    ROWCND: number,
    COLCND: number,
    AMAX: number): string {

        let func = emlapack.cwrap('dlaqge_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] R: DOUBLE PRECISION[M]
            'number', // [in] C: DOUBLE PRECISION[N]
            'number', // [in] ROWCND: DOUBLE PRECISION
            'number', // [in] COLCND: DOUBLE PRECISION
            'number', // [in] AMAX: DOUBLE PRECISION
            'number', // [out] EQUED: CHARACTER
        ]);

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pROWCND = emlapack._malloc(8);
        let pCOLCND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pEQUED = emlapack._malloc(1);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pROWCND, ROWCND, 'double');
        emlapack.setValue(pCOLCND, COLCND, 'double');
        emlapack.setValue(pAMAX, AMAX, 'double');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pR = emlapack._malloc(8 * (M));
        let aR = new Float64Array(emlapack.HEAPF64.buffer, pR, (M));
        aR.set(R.data, R.offset);

        let pC = emlapack._malloc(8 * (N));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (N));
        aC.set(C.data, C.offset);

        func(pM, pN, pA, pLDA, pR, pC, pROWCND, pCOLCND, pAMAX, pEQUED);
        return emlapack.getValue(pEQUED, 'i8');
}