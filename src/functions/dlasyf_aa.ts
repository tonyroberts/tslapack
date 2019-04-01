
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasyf_aa(UPLO: string,
    J1: number,
    M: number,
    NB: number,
    A: ndarray<number>,
    LDA: number,
    H: number,
    LDH: number): ndarray<number> {

        let func = emlapack.cwrap('dlasyf_aa_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] J1: INTEGER
            'number', // [in] M: INTEGER
            'number', // [in] NB: INTEGER
            'number', // [in,out] A: DOUBLE PRECISION[LDA,M]
            'number', // [in] LDA: INTEGER
            'number', // [out] IPIV: INTEGER[M]
            'number', // [in,out] H: DOUBLE PRECISION
            'number', // [in] LDH: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION
        ]);

        let pUPLO = emlapack._malloc(1);
        let pJ1 = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pNB = emlapack._malloc(4);
        let pLDA = emlapack._malloc(4);
        let pH = emlapack._malloc(8);
        let pLDH = emlapack._malloc(4);
        let pWORK = emlapack._malloc(8);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pJ1, J1, 'i32');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pNB, NB, 'i32');
        emlapack.setValue(pLDA, LDA, 'i32');
        emlapack.setValue(pH, H, 'double');
        emlapack.setValue(pLDH, LDH, 'i32');

        let pA = emlapack._malloc(8 * (LDA) * (M));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (LDA) * (M));
        aA.set(A.data, A.offset);

        let pIPIV = emlapack._malloc(4 * (M));
        let IPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (M));

        func(pUPLO, pJ1, pM, pNB, pA, pLDA, pIPIV, pH, pLDH, pWORK);
        return ndarray(IPIV, [(M)]);
}