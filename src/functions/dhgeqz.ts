
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dhgeqz(JOB: string,
    COMPQ: string,
    COMPZ: string,
    N: number,
    ILO: number,
    IHI: number,
    H: ndarray<number>,
    LDH: number,
    T: ndarray<number>,
    LDT: number,
    Q: ndarray<number>,
    LDQ: number,
    Z: ndarray<number>,
    LDZ: number): ndarray<number> {

        let func = emlapack.cwrap('dhgeqz_', null, [
            'number', // [in] JOB: CHARACTER
            'number', // [in] COMPQ: CHARACTER
            'number', // [in] COMPZ: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] ILO: INTEGER
            'number', // [in] IHI: INTEGER
            'number', // [in,out] H: DOUBLE PRECISION[LDH, N]
            'number', // [in] LDH: INTEGER
            'number', // [in,out] T: DOUBLE PRECISION[LDT, N]
            'number', // [in] LDT: INTEGER
            'number', // [out] ALPHAR: DOUBLE PRECISION[N]
            'number', // [out] ALPHAI: DOUBLE PRECISION[N]
            'number', // [out] BETA: DOUBLE PRECISION[N]
            'number', // [in,out] Q: DOUBLE PRECISION[LDQ, N]
            'number', // [in] LDQ: INTEGER
            'number', // [in,out] Z: DOUBLE PRECISION[LDZ, N]
            'number', // [in] LDZ: INTEGER
            'number', // [out] WORK: DOUBLE PRECISION[MAX(1,LWORK)]
            'number', // [in] LWORK: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let LWORK = -1;

        let pJOB = emlapack._malloc(1);
        let pCOMPQ = emlapack._malloc(1);
        let pCOMPZ = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pILO = emlapack._malloc(4);
        let pIHI = emlapack._malloc(4);
        let pLDH = emlapack._malloc(4);
        let pLDT = emlapack._malloc(4);
        let pLDQ = emlapack._malloc(4);
        let pLDZ = emlapack._malloc(4);
        let pLWORK = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOB, JOB.charCodeAt(0), 'i8');
        emlapack.setValue(pCOMPQ, COMPQ.charCodeAt(0), 'i8');
        emlapack.setValue(pCOMPZ, COMPZ.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pILO, ILO, 'i32');
        emlapack.setValue(pIHI, IHI, 'i32');
        emlapack.setValue(pLDH, LDH, 'i32');
        emlapack.setValue(pLDT, LDT, 'i32');
        emlapack.setValue(pLDQ, LDQ, 'i32');
        emlapack.setValue(pLDZ, LDZ, 'i32');
        emlapack.setValue(pLWORK, LWORK, 'i32');

        let pH = emlapack._malloc(8 * (LDH) * (N));
        let aH = new Float64Array(emlapack.HEAPF64.buffer, pH, (LDH) * (N));
        aH.set(H.data, H.offset);

        let pT = emlapack._malloc(8 * (LDT) * (N));
        let aT = new Float64Array(emlapack.HEAPF64.buffer, pT, (LDT) * (N));
        aT.set(T.data, T.offset);

        let pALPHAR = emlapack._malloc(8 * (N));
        let ALPHAR = new Float64Array(emlapack.HEAPF64.buffer, pALPHAR, (N));

        let pALPHAI = emlapack._malloc(8 * (N));
        let ALPHAI = new Float64Array(emlapack.HEAPF64.buffer, pALPHAI, (N));

        let pBETA = emlapack._malloc(8 * (N));
        let BETA = new Float64Array(emlapack.HEAPF64.buffer, pBETA, (N));

        let pQ = emlapack._malloc(8 * (LDQ) * (N));
        let aQ = new Float64Array(emlapack.HEAPF64.buffer, pQ, (LDQ) * (N));
        aQ.set(Q.data, Q.offset);

        let pZ = emlapack._malloc(8 * (LDZ) * (N));
        let aZ = new Float64Array(emlapack.HEAPF64.buffer, pZ, (LDZ) * (N));
        aZ.set(Z.data, Z.offset);

        let pWORK = emlapack._malloc(8 * (Math.max(1,LWORK)));
        let WORK = new Float64Array(emlapack.HEAPF64.buffer, pWORK, (Math.max(1,LWORK)));

        func(pJOB, pCOMPQ, pCOMPZ, pN, pILO, pIHI, pH, pLDH, pT, pLDT, pALPHAR, pALPHAI, pBETA, pQ, pLDQ, pZ, pLDZ, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dhgeqz': " + INFO);
        }

        LWORK = emlapack.getValue(pWORK, 'double');
        emlapack.setValue(pLWORK, LWORK, 'i32');
        pWORK = emlapack._malloc(8 * LWORK);

        func(pJOB, pCOMPQ, pCOMPZ, pN, pILO, pIHI, pH, pLDH, pT, pLDT, pALPHAR, pALPHAI, pBETA, pQ, pLDQ, pZ, pLDZ, pWORK, pLWORK, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dhgeqz': " + INFO);
        }

        return ndarray(BETA, [(N)]);
}