
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlarra(N: number,
    D: ndarray<number>,
    E: ndarray<number>,
    E2: ndarray<number>,
    SPLTOL: number,
    TNRM: number): ndarray<number> {

        let func = emlapack.cwrap('dlarra_', null, [
            'number', // [in] N: INTEGER
            'number', // [in] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N]
            'number', // [in,out] E2: DOUBLE PRECISION[N]
            'number', // [in] SPLTOL: DOUBLE PRECISION
            'number', // [in] TNRM: DOUBLE PRECISION
            'number', // [out] NSPLIT: INTEGER
            'number', // [out] ISPLIT: INTEGER[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pN = emlapack._malloc(4);
        let pSPLTOL = emlapack._malloc(8);
        let pTNRM = emlapack._malloc(8);
        let pNSPLIT = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pSPLTOL, SPLTOL, 'double');
        emlapack.setValue(pTNRM, TNRM, 'double');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N));
        aE.set(E.data, E.offset);

        let pE2 = emlapack._malloc(8 * (N));
        let aE2 = new Float64Array(emlapack.HEAPF64.buffer, pE2, (N));
        aE2.set(E2.data, E2.offset);

        let pISPLIT = emlapack._malloc(4 * (N));
        let ISPLIT = new Int32Array(emlapack.HEAPI32.buffer, pISPLIT, (N));

        func(pN, pD, pE, pE2, pSPLTOL, pTNRM, pNSPLIT, pISPLIT, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dlarra': " + INFO);
        }
        return ndarray(ISPLIT, [(N)]);
}