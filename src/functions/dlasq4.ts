
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasq4(I0: number,
    N0: number,
    Z: ndarray<number>,
    PP: number,
    N0IN: number,
    DMIN: number,
    DMIN1: number,
    DMIN2: number,
    DN: number,
    DN1: number,
    DN2: number,
    G: number): number {

        let func = emlapack.cwrap('dlasq4_', null, [
            'number', // [in] I0: INTEGER
            'number', // [in] N0: INTEGER
            'number', // [in] Z: DOUBLE PRECISION[4*N0]
            'number', // [in] PP: INTEGER
            'number', // [in] N0IN: INTEGER
            'number', // [in] DMIN: DOUBLE PRECISION
            'number', // [in] DMIN1: DOUBLE PRECISION
            'number', // [in] DMIN2: DOUBLE PRECISION
            'number', // [in] DN: DOUBLE PRECISION
            'number', // [in] DN1: DOUBLE PRECISION
            'number', // [in] DN2: DOUBLE PRECISION
            'number', // [out] TAU: DOUBLE PRECISION
            'number', // [out] TTYPE: INTEGER
            'number', // [in,out] G: DOUBLE PRECISION
        ]);

        let pI0 = emlapack._malloc(4);
        let pN0 = emlapack._malloc(4);
        let pPP = emlapack._malloc(4);
        let pN0IN = emlapack._malloc(4);
        let pDMIN = emlapack._malloc(8);
        let pDMIN1 = emlapack._malloc(8);
        let pDMIN2 = emlapack._malloc(8);
        let pDN = emlapack._malloc(8);
        let pDN1 = emlapack._malloc(8);
        let pDN2 = emlapack._malloc(8);
        let pTAU = emlapack._malloc(8);
        let pTTYPE = emlapack._malloc(4);
        let pG = emlapack._malloc(8);

        emlapack.setValue(pI0, I0, 'i32');
        emlapack.setValue(pN0, N0, 'i32');
        emlapack.setValue(pPP, PP, 'i32');
        emlapack.setValue(pN0IN, N0IN, 'i32');
        emlapack.setValue(pDMIN, DMIN, 'double');
        emlapack.setValue(pDMIN1, DMIN1, 'double');
        emlapack.setValue(pDMIN2, DMIN2, 'double');
        emlapack.setValue(pDN, DN, 'double');
        emlapack.setValue(pDN1, DN1, 'double');
        emlapack.setValue(pDN2, DN2, 'double');
        emlapack.setValue(pG, G, 'double');

        let pZ = emlapack._malloc(8 * (4*N0));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (4*N0));
        aZ.set(Z.data, Z.offset);

        func(pI0, pN0, pZ, pPP, pN0IN, pDMIN, pDMIN1, pDMIN2, pDN, pDN1, pDN2, pTAU, pTTYPE, pG);
        return emlapack.getValue(pTTYPE, 'i32');
}