
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpftrs(TRANSR: string,
    UPLO: string,
    N: number,
    NRHS: number,
    A: ndarray<number>,
    B: ndarray<number>,
    LDB: number) {

        let func = emlapack.cwrap('dpftrs_', null, [
            'number', // [in] TRANSR: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] NRHS: INTEGER
            'number', // [in] A: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [in,out] B: DOUBLE PRECISION[LDB,NRHS]
            'number', // [in] LDB: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pTRANSR = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pNRHS = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTRANSR, TRANSR.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNRHS, NRHS, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');

        let pA = emlapack._malloc(8 * (N*(N+1)/2));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (N*(N+1)/2));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (NRHS));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (NRHS));
        aB.set(B.data, B.offset);

        func(pTRANSR, pUPLO, pN, pNRHS, pA, pB, pLDB, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpftrs': " + INFO);
        }
}