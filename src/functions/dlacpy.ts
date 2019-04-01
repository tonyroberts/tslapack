
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlacpy(UPLO: string,
    M: number,
    N: number,
    A: ndarray<number>,
    LDA: number,
    LDB: number): ndarray<number> {

        let func = emlapack.cwrap('dlacpy_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [out] B: DOUBLE PRECISION[LDB,N]
            'number', // [in] LDB: INTEGER
        ]);

        let pUPLO = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pLDB = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pLDB, LDB, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pB = emlapack._malloc(8 * (LDB) * (N));
        let B = new Float64Array(emlapack.HEAPF64.buffer, pB, (LDB) * (N));

        func(pUPLO, pM, pN, pA, pLDA, pB, pLDB);
        return ndarray(B, [(LDB), (N)]);
}