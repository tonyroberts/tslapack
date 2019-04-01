import * as ndarray from 'ndarray';
import {dsyev} from "../tslapack";

test("dseyev", () => {

    let A = ndarray(new Float64Array([
              1.96, 0, 0, 0, 0,
             -6.49, 3.8, 0, 0, 0,
             -0.47, -6.39, 4.17, 0, 0,
             -7.2, 1.5, -1.51, 5.7, 0,
             -0.65, -6.34, 2.67, 1.8, -7.1]),
        [5, 5]);

    let result = dsyev('V', 'U', 5, A, 5);

    function round(x: number): number {
        return Number(x.toPrecision(6));
    }

    expect(result.data.map<number>(round)).toEqual(new Float64Array([
        -11.065575263268391,
        -6.22874693239854,
         0.8640279752720624,
         8.865457108365518,
         16.09483711202934
    ].map(round)));
});
