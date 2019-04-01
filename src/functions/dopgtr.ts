
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dopgtr(UPLO: string,
    N: number,
    AP: ndarray<number>,
    TAU: ndarray<number>,
    LDQ: number): ndarray<number> {

        let func = emlapack.cwrap('dopgtr_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [in] TAU: DOUBLE PRECISION[N-1]
            'number', // [out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[N-1]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pTAU = emlapack._malloc(8 * (N-1));
        let aTAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (N-1));
        aTAU.set(TAU.data, TAU.offset);

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let Q = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));

        let pWORK = emlapack._malloc(8 * (N-1));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N-1));

        func(pUPLO, pN, pAP, pTAU, pQ, pLDQ, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dopgtr': " + INFO);
        }
        return ndarray(Q, [(LDQ), (N)]);
}