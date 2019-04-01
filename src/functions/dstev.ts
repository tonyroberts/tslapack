
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dstev(JOBZ: string,
    N: number,
    D: ndarray<number>,
    E: ndarray<number>,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dstev_', null, [
            'number', // [in] JOBZ: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N-1]
            'number', // [out] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[max(1,2*N-2)]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOBZ = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOBZ, JOBZ.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let Z = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));

        let pWORK = emlapack._malloc(8 * (Math.max(1,2*N-2)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,2*N-2)));

        func(pJOBZ, pN, pD, pE, pZ, pLDZ, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dstev': " + INFO);
        }
        return ndarray(Z, [(LDZ), (N)]);
}