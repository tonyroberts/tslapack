
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlascl(TYPE: string,
    KL: number,
    KU: number,
    CFROM: number,
    CTO: number,
    M: number,
    N: number,
    A: ndarray<number>,
    LDA: number) {

        let func = emlapack.cwrap('dlascl_', null, [
            'number', // [in] TYPE: CHARACTER
            'number', // [in] KL: INTEGER
            'number', // [in] KU: INTEGER
            'number', // [in] CFROM: DOUBLE PRECISION
            'number', // [in] CTO: DOUBLE PRECISION
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pTYPE = emlapack._malloc(1);
        let pKL = emlapack._malloc(4);
        let pKU = emlapack._malloc(4);
        let pCFROM = emlapack._malloc(8);
        let pCTO = emlapack._malloc(8);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pTYPE, TYPE.charCodeAt(0), 'i8');
        emlapack.setValue(pKL, KL, 'i32');
        emlapack.setValue(pKU, KU, 'i32');
        emlapack.setValue(pCFROM, CFROM, 'double');
        emlapack.setValue(pCTO, CTO, 'double');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        func(pTYPE, pKL, pKU, pCFROM, pCTO, pM, pN, pA, pLDA, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlascl': " + INFO);
        }
}