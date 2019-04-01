
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dtrexc(COMPQ: string,
    N: number,
    T: ndarray<number>,
    LDT: number,
    Q: ndarray<number>,
    LDQ: number,
    IFST: number,
    ILST: number) {

        let func = emlapack.cwrap('dtrexc_', null, [
            'number', // [in] COMPQ: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] T: DOUBLE PRECISION[LDT,N]
            'number', // [in] LDT: INTEGER
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ,N]
            'number', // [in] LDQ: INTEGER
            'number', // [in,out] IFST: INTEGER
            'number', // [in,out] ILST: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[N]
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pCOMPQ = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pLDT = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pIFST = emlapack._malloc(4);
        let pILST = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pCOMPQ, COMPQ.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pLDT, LDT, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pIFST, IFST, 'i32');
        emlapack.setValue(pILST, ILST, 'i32');

        let pT = emlapack._malloc(8 * (LDT) * (N));
        let aT = new Float64Array(emlapack.HEAPF64.buffer, pT, (LDT) * (N));
        aT.set(T.data, T.offset);

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let aQ = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));
        aQ.set(Q.data, Q.offset);

        let pWORK = emlapack._malloc(8 * (N));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (N));

        func(pCOMPQ, pN, pT, pLDT, pQ, pLDQ, pIFST, pILST, pWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dtrexc': " + INFO);
        }
}