
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgbtf2(M: number,
    N: number,
    KL: number,
    KU: number,
    AB: ndarray<number>,
    LDAB: number): ndarray<number> {

        let func = emlapack.cwrap('dgbtf2_', null, [
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] KL: INTEGER
            'number', // [in] KU: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] IPIV: INTEGER[min(M,N)]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pKL = emlapack._malloc(4);
        let pKU = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pKL, KL, 'i32');
        emlapack.setValue(pKU, KU, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pIPIV = emlapack._malloc(4 * (Math.min(M,N)));
        let IPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (Math.min(M,N)));

        func(pM, pN, pKL, pKU, pAB, pLDAB, pIPIV, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgbtf2': " + INFO);
        }
        return ndarray(IPIV, [(Math.min(M,N))]);
}