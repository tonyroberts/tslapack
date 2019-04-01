
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dsptrd(UPLO: string,
    N: number,
    AP: ndarray<number>): ndarray<number> {

        let func = emlapack.cwrap('dsptrd_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [out] D: DOUBLE PRECISION[N]
            'number', // [out] E: DOUBLE PRECISION[N-1]
            'number', // [out] TAU: DOUBLE PRECISION[N-1]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pD = emlapack._malloc(8 * (N));
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));

        let pE = emlapack._malloc(8 * (N-1));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));

        let pTAU = emlapack._malloc(8 * (N-1));
        let TAU = new Float64Array(emlapack.HEAPF64.buffer, pTAU, (N-1));

        func(pUPLO, pN, pAP, pD, pE, pTAU, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dsptrd': " + INFO);
        }
        return ndarray(TAU, [(N-1)]);
}