
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlamrg(N1: number,
    N2: number,
    A: ndarray<number>,
    DTRD1: number,
    DTRD2: number): ndarray<number> {

        let func = emlapack.cwrap('dlamrg_', null, [
            'number', // [in] N1: INTEGER
            'number', // [in] N2: INTEGER
            'number', // [in] A: DOUBLE PRECISION[N1+N2]
            'number', // [in] DTRD1: INTEGER
            'number', // [in] DTRD2: INTEGER
            'number', // [out] INDEX: INTEGER[N1+N2]
        ]);

        let pN1 = emlapack._malloc(4);
        let pN2 = emlapack._malloc(4);
        let pDTRD1 = emlapack._malloc(4);
        let pDTRD2 = emlapack._malloc(4);

        emlapack.setValue(pN1, N1, 'i32');
        emlapack.setValue(pN2, N2, 'i32');
        emlapack.setValue(pDTRD1, DTRD1, 'i32');
        emlapack.setValue(pDTRD2, DTRD2, 'i32');

        let pA = emlapack._malloc(8 * (N1+N2));
        let aA = new Float64Array(emlapack.HEAPF64.buffer, pA, (N1+N2));
        aA.set(A.data, A.offset);

        let pINDEX = emlapack._malloc(4 * (N1+N2));
        let INDEX = new Int32Array(emlapack.HEAPI32.buffer, pINDEX, (N1+N2));

        func(pN1, pN2, pA, pDTRD1, pDTRD2, pINDEX);
        return ndarray(INDEX, [(N1+N2)]);
}