
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtgsy2(TRANS: string,
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
    LDF: number,
    RDSUM: number,
    RDSCAL: number): number {

        let func = emlapack.cwrap('dtgsy2_', null, [
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
            'number', // [out] SCALE: DOUBLE PRECISION
            'number', // [in,out] RDSUM: DOUBLE PRECISION
            'number', // [in,out] RDSCAL: DOUBLE PRECISION
            'number', // [out] IWORK: INTEGER[M+N+2]
            'number', // [out] PQ: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
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
        let pSCALE = emlapack._malloc(8);
        let pRDSUM = emlapack._malloc(8);
        let pRDSCAL = emlapack._malloc(8);
        let pPQ = emlapack._malloc(4);
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
        emlapack.setValue(pRDSUM, RDSUM, 'double');
        emlapack.setValue(pRDSCAL, RDSCAL, 'double');

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

        let pIWORK = emlapack._malloc(4 * (M+N+2));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (M+N+2));

        func(pTRANS, pIJOB, pM, pN, pA, pLDA, pB, pLDB, pC, pLDC, pD, pLDD, pE, pLDE, pF, pLDF, pSCALE, pRDSUM, pRDSCAL, pIWORK, pPQ, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtgsy2': " + INFO);
        }
        return emlapack.getValue(pPQ, 'i32');
}