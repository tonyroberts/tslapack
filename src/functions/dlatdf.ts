
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlatdf(IJOB: number,
    N: number,
    Z: ndarray<number>,
    LDZ: number,
    RHS: ndarray<number>,
    RDSUM: number,
    RDSCAL: number,
    IPIV: ndarray<number>,
    JPIV: ndarray<number>) {

        let func = emlapack.cwrap('dlatdf_', null, [
            'number', // [in] IJOB: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [in,out] RHS: DOUBLE PRECISION[N]
            'number', // [in,out] RDSUM: DOUBLE PRECISION
            'number', // [in,out] RDSCAL: DOUBLE PRECISION
            'number', // [in] IPIV: INTEGER[N]
            'number', // [in] JPIV: INTEGER[N]
        ]);

        let pIJOB = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pRDSUM = emlapack._malloc(8);
        let pRDSCAL = emlapack._malloc(8);

        emlapack.setValue(pIJOB, IJOB, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');
        emlapack.setValue(pRDSUM, RDSUM, 'double');
        emlapack.setValue(pRDSCAL, RDSCAL, 'double');

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));
        aZ.set(Z.data, Z.offset);

        let pRHS = emlapack._malloc(8 * (N));
        let aRHS = new Float64Array(emlapack.HEAPF64.buffer, pRHS, (N));
        aRHS.set(RHS.data, RHS.offset);

        let pIPIV = emlapack._malloc(4 * (N));
        let aIPIV = new Int32Array(emlapack.HEAPI32.buffer, pIPIV, (N));
        aIPIV.set(IPIV.data, IPIV.offset);

        let pJPIV = emlapack._malloc(4 * (N));
        let aJPIV = new Int32Array(emlapack.HEAPI32.buffer, pJPIV, (N));
        aJPIV.set(JPIV.data, JPIV.offset);

        func(pIJOB, pN, pZ, pLDZ, pRHS, pRDSUM, pRDSCAL, pIPIV, pJPIV);
}