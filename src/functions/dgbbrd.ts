
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgbbrd(VECT: string,
    M: number,
    N: number,
    NCC: number,
    KL: number,
    KU: number,
    AB: ndarray<number>,
    LDAB: number,
    LDQ: number,
    LDPT: number,
    C: ndarray<number>,
    LDC: number): ndarray<number> {

        let func = emlapack.cwrap('dgbbrd_', null, [
            'number', // [in] VECT: CHARACTER
            'number', // [in] M: INTEGER
            'number', // [in] N: INTEGER
            'number', // [in] NCC: INTEGER
            'number', // [in] KL: INTEGER
            'number', // [in] KU: INTEGER
            'number', // [in,out] AB: DOUBLE PRECISION[LDAB,N]
            'number', // [in] LDAB: INTEGER
            'number', // [out] D: DOUBLE PRECISION[min(M,N)]
            'number', // [out] E: DOUBLE PRECISION[min(M,N)-1]
            'number', // [out] Q: DOUBLE PRECISION[LDQ,M]
            'number', // [in] LDQ: INTEGER
            'number', // [out] PT: DOUBLE PRECISION[LDPT,N]
            'number', // [in] LDPT: INTEGER
            'number', // [in,out] C: DOUBLE PRECISION[LDC,NCC]
            'number', // [in] LDC: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[2*max(M,N)]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pVECT = emlapack._malloc(1);
        let pM = emlapack._malloc(4);
        let pN = emlapack._malloc(4);
        let pNCC = emlapack._malloc(4);
        let pKL = emlapack._malloc(4);
        let pKU = emlapack._malloc(4);
        let pLDAB = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pLDPT = emlapack._malloc(4);
        let pLDC = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pVECT, VECT.charCodeAt(0), 'i8');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pNCC, NCC, 'i32');
        emlapack.setValue(pKL, KL, 'i32');
        emlapack.setValue(pKU, KU, 'i32');
        emlapack.setValue(pLDAB, LDAB, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pLDPT, LDPT, 'i32');
        emlapack.setValue(pLDC, LDC, 'i32');

        let pAB = emlapack._malloc(8 * (LDAB) * (N));
        let aAB = new Float64Array(emlapack.HEAPF64.buffer, pAB, (LDAB) * (N));
        aAB.set(AB.data, AB.offset);

        let pD = emlapack._malloc(8 * (Math.min(M,N)));
        let D = new Float64Array(emlapack.HEAPF64.buffer, pD, (Math.min(M,N)));

        let pE = emlapack._malloc(8 * (Math.min(M,N)-1));
        let E = new Float64Array(emlapack.HEAPF64.buffer, pE, (Math.min(M,N)-1));

        let pQ = emlapack._malloc(8 * (LDQ) * (M));
        let Q = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (M));

        let pPT = emlapack._malloc(8 * (LDPT) * (N));
        let PT = new Float64Array(emlapack.HEAPF64.buffer, pPT, (LDPT) * (N));

        let pC = emlapack._malloc(8 * (LDC) * (NCC));
        let aC = new Float64Array(emlapack.HEAPF64.buffer, pC, (LDC) * (NCC));
        aC.set(C.data, C.offset);

        let pWORK = emlapack._malloc(8 * (2*Math.max(M,N)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (2*Math.max(M,N)));

        func(pVECT, pM, pN, pNCC, pKL, pKU, pAB, pLDAB, pD, pE, pQ, pLDQ, pPT, pLDPT, pC, pLDC, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgbbrd': " + INFO);
        }
        return ndarray(PT, [(LDPT), (N)]);
}