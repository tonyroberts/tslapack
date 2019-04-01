
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtrsyl(TRANA: string,
    TRANB: string,
    ISGN: number,
    M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    C: ndarray<number>,
    LDC: number): number {

        let func = emlapack.cwrap('dtrsyl_', null, [
            'number', // [in] TRANA: CHARACTER
            'number', // [in] TRANB: CHARACTER
            'number', // [in] ISGN: INTEGER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,M]
            'number', // [in] LDA: INTEGER
            'number', // [in] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
            'number', // [in,out] C: DOUBLE PRECISION[LDC,N]
            'number', // [in] LDC: INTEGER
            'number', // [out] SCALE: DOUBLE PRECISION
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pTRANA = emlapack._malloc(1);
        let pTRANB = emlapack._malloc(1);
        let pISGN = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDC = emlapack._malloc(4);
        let pSCALE = emlapack._malloc(8);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANA, TRANA.charCodeAt(0), 'i8');
        emlapack.setValue(pTRANB, TRANB.charCodeAt(0), 'i8');
        emlapack.setValue(pISGN, ISGN, 'i32');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDC, LDC, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (M));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (M));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pC = emlapack._malloc(8 * (LDC) * (N));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (LDC) * (N));
        aC.set(C.data, C.offset);

        func(pTRANA, pTRANB, pISGN, pM, pN, pA, pLDA, pB, pLDB, pC, pLDC, pSCALE, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtrsyl': " + INFO);
        }
        return emlapack.getValue(pSCALE, 'double');
}