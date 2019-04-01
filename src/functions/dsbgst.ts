
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsbgst(VECT: string,
    UPLO: string,
    N: number,
    KA: number,
    KB: number,
    AB: ndarray<number>,
    LDAB: number,
    BB: ndarray<number>,
    LDBB: number,
    LDX: number): ndarray<number> {

        let func = emlapack.cwrap('dsbgst_', null, [
            'number', // [in] VECT: CHARACTER
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] KA: INTEGER
            'number', // [in] KB: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [in] BB: DOUBLE PRECISION[LDBB,N]
            'number', // [in] LDBB: INTEGER
            'number', // [out] X: DOUBLE PRECISION[LDX,N]
            'number', // [in] LDX: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[2*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pVECT = emlapack._malloc(1);
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pKA = emlapack._malloc(4);
        let pKB = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLDBB = emlapack._malloc(4);
        let pLDX = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pVECT, VECT.charCodeAt(0), 'i8');
        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKA, KA, 'i32');
        emlapack.setValue(pKB, KB, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLDBB, LDBB, 'i32');
        emlapack.setValue(pLDX, LDX, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pBB = emlapack._malloc(8 * (LDBB) * (N));
        let aBB = new Float64Array(emlapack.HEAPF64.buffer, pBB, (LDBB) * (N));
        aBB.set(BB.data, BB.offset);

        let pX = emlapack._malloc(8 * (LDX) * (N));
        let X = new Float64Array(emlapack.HEAPF64.buffer, pX, (LDX) * (N));

        let pWORK = emlapack._malloc(8 * (2*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*N));

        func(pVECT, pUPLO, pN, pKA, pKB, pAB, pLDAB, pBB, pLDBB, pX, pLDX, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsbgst': " + INFO);
        }
        return ndarray(X, [(LDX), (N)]);
}