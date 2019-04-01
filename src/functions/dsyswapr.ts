
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsyswapr(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    I1: number,
    I2: number) {

        let func = emlapack.cwrap('dsyswapr_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] I1: INTEGER
            'number', // [in] I2: INTEGER
        ]);

        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pI1 = emlapack._malloc(4);
        let pI2 = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pI1, I1, 'i32');
        emlapack.setValue(pI2, I2, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        func(pUPLO, pN, pA, pLDA, pI1, pI2);
}