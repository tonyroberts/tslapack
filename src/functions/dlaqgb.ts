
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaqgb(M: number,
    N: number,
    KL: number,
    KU: number,
    AB: ndarray<number>,
    LDAB: number,
    R: ndarray<number>,
    C: ndarray<number>,
    ROWCND: number,
    COLCND: number,
    AMAX: number): string {

        let func = emlapack.cwrap('dlaqgb_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] KL: INTEGER
            'number', // [in] KU: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [in] R: DOUBLE PRECISION[M]
            'number', // [in] C: DOUBLE PRECISION[N]
            'number', // [in] ROWCND: DOUBLE PRECISION
            'number', // [in] COLCND: DOUBLE PRECISION
            'number', // [in] AMAX: DOUBLE PRECISION
            'number', // [out] EQUED: CHARACTER
        ]);

        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pKL = emlapack._malloc(4);
        let pKU = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pROWCND = emlapack._malloc(8);
        let pCOLCND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pEQUED = emlapack._malloc(1);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKL, KL, 'i32');
        emlapack.setValue(pKU, KU, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pROWCND, ROWCND, 'double');
        emlapack.setValue(pCOLCND, COLCND, 'double');
        emlapack.setValue(pAMAX, AMAX, 'double');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pR = emlapack._malloc(8 * (M));
        let aR = new Float64Array(emlapack.HEAPF64.buffer, pR, (M));
        aR.set(R.data, R.offset);

        let pC = emlapack._malloc(8 * (N));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (N));
        aC.set(C.data, C.offset);

        func(pM, pN, pKL, pKU, pAB, pLDAB, pR, pC, pROWCND, pCOLCND, pAMAX, pEQUED);
        return emlapack.getValue(pEQUED, 'i8');
}