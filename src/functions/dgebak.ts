
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dgebak(JOB: string,
    SIDE: string,
    N: number,
    ILO: number,
    IHI: number,
    SCALE: ndarray<number>,
    M: number,
    V: ndarray<number>,
    LDV: number) {

        let func = emlapack.cwrap('dgebak_', null, [
            'number', // [in] JOB: CHARACTER
            'number', // [in] SIDE: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in] ILO: INTEGER
            'number', // [in] IHI: INTEGER
            'number', // [in] SCALE: DOUBLE PRECISION[N]
            'number', // [in] M: INTEGER
            'number', // [in,out] V: DOUBLE PRECISION[LDV,M]
            'number', // [in] LDV: INTEGER
            'number', // [out] INFO: INTEGER
        ]);

        let INFO;
        let pJOB = emlapack._malloc(1);
        let pSIDE = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pILO = emlapack._malloc(4);
        let pIHI = emlapack._malloc(4);
        let pM = emlapack._malloc(4);
        let pLDV = emlapack._malloc(4);
        let pINFO = emlapack._malloc(4);

        emlapack.setValue(pJOB, JOB.charCodeAt(0), 'i8');
        emlapack.setValue(pSIDE, SIDE.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pILO, ILO, 'i32');
        emlapack.setValue(pIHI, IHI, 'i32');
        emlapack.setValue(pM, M, 'i32');
        emlapack.setValue(pLDV, LDV, 'i32');

        let pSCALE = emlapack._malloc(8 * (N));
        let aSCALE = new Float64Array(emlapack.HEAPF64.buffer, pSCALE, (N));
        aSCALE.set(SCALE.data, SCALE.offset);

        let pV = emlapack._malloc(8 * (LDV) * (M));
        let aV = new Float64Array(emlapack.HEAPF64.buffer, pV, (LDV) * (M));
        aV.set(V.data, V.offset);

        func(pJOB, pSIDE, pN, pILO, pIHI, pSCALE, pM, pV, pLDV, pINFO);
        INFO = emlapack.getValue(pINFO, 'i32');
        if (INFO != 0) {
            throw new Error("Error calling 'dgebak': " + INFO);
        }
}