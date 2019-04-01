
import * as ndarray from 'ndarray';
let emlapack = require('emlapack');

export function dlaqsp(UPLO: string,
    N: number,
    AP: ndarray<number>,
    S: ndarray<number>,
    SCOND: number,
    AMAX: number): string {

        let func = emlapack.cwrap('dlaqsp_', null, [
            'number', // [in] UPLO: CHARACTER
            'number', // [in] N: INTEGER
            'number', // [in,out] AP: DOUBLE PRECISION[N*(N+1)/2]
            'number', // [in] S: DOUBLE PRECISION[N]
            'number', // [in] SCOND: DOUBLE PRECISION
            'number', // [in] AMAX: DOUBLE PRECISION
            'number', // [out] EQUED: CHARACTER
        ]);

        let pUPLO = emlapack._malloc(1);
        let pN = emlapack._malloc(4);
        let pSCOND = emlapack._malloc(8);
        let pAMAX = emlapack._malloc(8);
        let pEQUED = emlapack._malloc(1);

        emlapack.setValue(pUPLO, UPLO.charCodeAt(0), 'i8');
        emlapack.setValue(pN, N, 'i32');
        emlapack.setValue(pSCOND, SCOND, 'double');
        emlapack.setValue(pAMAX, AMAX, 'double');

        let pAP = emlapack._malloc(8 * (N*(N+1)/2));
        let aAP = new Float64Array(emlapack.HEAPF64.buffer, pAP, (N*(N+1)/2));
        aAP.set(AP.data, AP.offset);

        let pS = emlapack._malloc(8 * (N));
        let aS = new Float64Array(emlapack.HEAPF64.buffer, pS, (N));
        aS.set(S.data, S.offset);

        func(pUPLO, pN, pAP, pS, pSCOND, pAMAX, pEQUED);
        return emlapack.getValue(pEQUED, 'i8');
}