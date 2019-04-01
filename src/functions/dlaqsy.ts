
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaqsy(UPLO: string,
    N: number,
    A: ndarray<number>,
    LDA: number,
    S: ndarray<number>,
    SCOND: number,
    AMAX: number): string {

        let func = emlapack.cwrap('dlaqsy_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,N]
            'number', // [in] LDA: INTEGER
            'number', // [in] S: DOUBLE PRECISION[N]
            'number', // [in] SCOND: DOUBLE PRECISION
            'number', // [in] AMAX: DOUBLE PRECISION
            'number', // [out] EQUED: CHARACTER
        ]);

        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pSCOND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pEQUED = emlapack._malloc(1);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pSCOND, SCOND, 'double');
        emlapack.setValue(pAMAX, AMAX, 'double');

        let pA = emlapack._malloc(8 * (LDA) * (N));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (N));
        aA.set(A.data, A.offset);

        let pS = emlapack._malloc(8 * (N));
        let aS = new Float64Array(emlapack.HEAPF64.buffer, pS, (N));
        aS.set(S.data, S.offset);

        func(pUPLO, pN, pA, pLDA, pS, pSCOND, pAMAX, pEQUED);
        return emlapack.getValue(pEQUED, 'i8');
}