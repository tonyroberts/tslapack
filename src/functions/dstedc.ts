
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dstedc(COMPZ: string,
    N: number,
    D: ndarray<number>,
    E: ndarray<number>,
    Z: ndarray<number>,
    LDZ: number) {

        let func = emlapack.cwrap('dstedc_', null, [
            'number', // [in] COMPZ: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] D: DOUBLE PRECISION[N]
            'number', // [in,out] E: DOUBLE PRECISION[N-1]
            'number', // [in,out] Z: DOUBLE PRECISION[LDZ,N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] IWORK: INTEGER[MAX(1,LIWORK)]
            'number', // [in] LIWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;
        let LIWORK = -1;

        let pCOMPZ = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pLIWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pCOMPZ, COMPZ.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        emlapack.setValue(pLIWORK, LIWORK, 'i32');

        let pD = emlapack._malloc(8 * (N));
        let aD = new Float64Array(emlapack.HEAPF64.buffer, pD, (N));
        aD.set(D.data, D.offset);

        let pE = emlapack._malloc(8 * (N-1));
        let aE = new Float64Array(emlapack.HEAPF64.buffer, pE, (N-1));
        aE.set(E.data, E.offset);

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));
        aZ.set(Z.data, Z.offset);

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        let pIWORK = emlapack._malloc(4 * (Math.max(1,LIWORK)));
        let IWORK = new Int32Array(emlapack.HEAPI32.buffer, pIWORK, (Math.max(1,LIWORK)));

        func(pCOMPZ, pN, pD, pE, pZ, pLDZ, pWORK, pLWORK, pIWORK, pLIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dstedc': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);
        LIWORK = emlapack.getValue(pIWORK, 'i32');
        emlapack.setValue(pLIWORK, LIWORK, 'i32');
        pIWORK = emlapack._malloc(4 * LIWORK);

        func(pCOMPZ, pN, pD, pE, pZ, pLDZ, pWORK, pLWORK, pIWORK, pLIWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dstedc': " + INFO);
        }

}