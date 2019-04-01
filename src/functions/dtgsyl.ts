
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtgsyl(TRANS: string,
    IJOB: number,
    M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    C: ndarray<number>,
    LDC: number,
    D: ndarray<number>,
    LDD: number,
    E: ndarray<number>,
    LDE: number,
    F: ndarray<number>,
    LDF: number): number {

        let func = emlapack.cwrap('dtgsyl_', null, [
            'number', // [in] TRANS: CHARACTER
            'number', // [in] IJOB: INTEGER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA, M]
            'number', // [in] LDA: INTEGER
            'number', // [in] B: DOUBLE PRECISION[LDB, N]
            'number', // [in] LDB: INTEGER
            'number', // [in,out] C: DOUBLE PRECISION[LDC, N]
            'number', // [in] LDC: INTEGER
            'number', // [in] D: DOUBLE PRECISION[LDD, M]
            'number', // [in] LDD: INTEGER
            'number', // [in] E: DOUBLE PRECISION[LDE, N]
            'number', // [in] LDE: INTEGER
            'number', // [in,out] F: DOUBLE PRECISION[LDF, N]
            'number', // [in] LDF: INTEGER
            'number', // [out] DIF: DOUBLE PRECISION
            'number', // [out] SCALE: DOUBLE PRECISION
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] IWORK: INTEGER[M+N+6]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pTRANS = emlapack._malloc(1);
        let pIJOB = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDC = emlapack._malloc(4);
        let pLDD = emlapack._malloc(4);
        let pLDE = emlapack._malloc(4);
        let pLDF = emlapack._malloc(4);
        let pDIF = emlapack._malloc(8);
        let pSCALE = emlapack._malloc(8);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANS, TRANS.charCodeAt(0), 'i8');
        emlapack.setValue(pIJOB, IJOB, 'i32');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDC, LDC, 'i32');
        emlapack.setValue(pLDD, LDD, 'i32');
        emlapack.setValue(pLDE, LDE, 'i32');
        emlapack.setValue(pLDF, LDF, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (M));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (M));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pC = emlapack._malloc(8 * (LDC) * (N));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (LDC) * (N));
        aC.set(C.data, C.offset);

        let pD = emlapack._malloc(8 * (LDD) * (M));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (LDD) * (M));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (LDE) * (N));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (LDE) * (N));
        aE.set(E.data, E.offset);

        let pF = emlapack._malloc(8 * (LDF) * (N));
        let aF = new Float64Array(emlapack.HEAPF64.buffer, pF, (LDF) * (N));
        aF.set(F.data, F.offset);

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        let pIWORK = emlapack._malloc(4 * (M+N+6));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (M+N+6));

        func(pTRANS, pIJOB, pM, pN, pA, pLDA, pB, pLDB, pC, pLDC, pD, pLDD, pE, pLDE, pF, pLDF, pDIF, pSCALE, pWORK, pLWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtgsyl': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pTRANS, pIJOB, pM, pN, pA, pLDA, pB, pLDB, pC, pLDC, pD, pLDD, pE, pLDE, pF, pLDF, pDIF, pSCALE, pWORK, pLWORK, pIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtgsyl': " + INFO);
        }

        return emlapack.getValue(pSCALE, 'double');
}