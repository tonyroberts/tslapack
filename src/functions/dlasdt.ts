
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlasdt(N: number,
    MSUB: number): ndarray<number> {

        let func = emlapack.cwrap('dlasdt_', null, [
            'number', // [in] N: INTEGER
            'number', // [out] LVL: INTEGER
            'number', // [out] ND: INTEGER
            'number', // [out] INODE: INTEGER[N]
            'number', // [out] NDIML: INTEGER[N]
            'number', // [out] NDIMR: INTEGER[N]
            'number', // [in] MSUB: INTEGER
        ]);

        let pN = emlapack._malloc(4);
        let pLVL = emlapack._malloc(4);
        let pND = emlapack._malloc(4);
        let pMSUB = emlapack._malloc(4);

        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pMSUB, MSUB, 'i32');

        let pINODE = emlapack._malloc(4 * (N));
        let INODE = new Int32Array(emlapack.HEAPI32.buffer, pINODE, (N));

        let pNDIML = emlapack._malloc(4 * (N));
        let NDIML = new Int32Array(emlapack.HEAPI32.buffer, pNDIML, (N));

        let pNDIMR = emlapack._malloc(4 * (N));
        let NDIMR = new Int32Array(emlapack.HEAPI32.buffer, pNDIMR, (N));

        func(pN, pLVL, pND, pINODE, pNDIML, pNDIMR, pMSUB);
        return ndarray(NDIMR, [(N)]);
}