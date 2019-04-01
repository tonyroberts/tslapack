
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dpteqr(COMPZ: string,
    N: number,
    D: ndarray<number>,
    E: ndarray<number>,
    Z: ndarray<number>,
    LDZ: number) {

        let func = emlapack.cwrap('dpteqr_', null, [
            'number', // [in] COMPZ: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N-1]
            'number', // [in,out] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[4*N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pCOMPZ = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pCOMPZ, COMPZ.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));
        aZ.set(Z.data, Z.offset);

        let pWORK = emlapack._malloc(8 * (4*N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (4*N));

        func(pCOMPZ, pN, pD, pE, pZ, pLDZ, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dpteqr': " + INFO);
        }
}