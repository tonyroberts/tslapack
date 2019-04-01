
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgghrd(COMPQ: string,
    COMPZ: string,
    N: number,
    ILO: number,
    IHI: number,
    A: ndarray<number>,
    LDA: number,
    B: ndarray<number>,
    LDB: number,
    Q: ndarray<number>,
    LDQ: number,
    Z: ndarray<number>,
    LDZ: number) {

        let func = emlapack.cwrap('dgghrd_', null, [
            'number', // [in] COMPQ: CHARACTER
            'number', // [in] COMPZ: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] ILO: INTEGER
            'number', // [in] IHI: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA, N]
            'number', // [in] LDA: INTEGER
            'number', // [in,out] B: DOUBLE PRECISION[LDB, N]
            'number', // [in] LDB: INTEGER
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ, N]
            'number', // [in] LDQ: INTEGER
            'number', // [in,out] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pCOMPQ = emlapack._malloc(1);
        let pCOMPZ = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pILO = emlapack._malloc(4);
        let pIHI = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pCOMPQ, COMPQ.charCodeAt(0), 'i8');
        emlapack.setValue(pCOMPZ, COMPZ.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pILO, ILO, 'i32');
        emlapack.setValue(pIHI, IHI, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let aB = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));
        aB.set(B.data, B.offset);

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let aQ = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));
        aQ.set(Q.data, Q.offset);

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));
        aZ.set(Z.data, Z.offset);

        func(pCOMPQ, pCOMPZ, pN, pILO, pIHI, pA, pLDA, pB, pLDB, pQ, pLDQ, pZ, pLDZ, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgghrd': " + INFO);
        }
}