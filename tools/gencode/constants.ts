export const FortranTypes = new Map<string, string>(Object.entries({
    'CHARACTER': 'string',
    'INTEGER': 'number',
    'DOUBLE PRECISION': 'number',
    'DOUBLE': 'number',
    'REAL': 'number',
}));

export const SizeOf = new Map<string, number>(Object.entries({
    'CHARACTER': 1,
    'INTEGER': 4,
    'DOUBLE PRECISION': 8,
    'DOUBLE': 8,
    'REAL': 8,
}));

export const EmLapackTypes = new Map<string, string>(Object.entries({
    'CHARACTER': 'i8',
    'INTEGER': 'i32',
    'DOUBLE PRECISION': 'double',
    'DOUBLE': 'double',
    'REAL': 'double',
}));

export const EmLapackModifiers = new Map<string, (x: string) => string>(Object.entries({
    'CHARACTER': (s: string): string => s + '.charCodeAt(0)'
}));

export const ArrayTypes = new Map<string, string>(Object.entries({
    'INTEGER': 'Int32Array',
    'DOUBLE PRECISION': 'Float64Array',
    'DOUBLE': 'Float64Array',
    'REAL': 'Float64Array',
}));

export const HeapTypes = new Map<string, string>(Object.entries({
    'INTEGER': 'HEAPI32',
    'DOUBLE PRECISION': 'HEAPF64',
    'DOUBLE': 'HEAPF64',
    'REAL': 'HEAPF64',
}));

export const UnsupportedTypes = new Map<string, boolean>(Object.entries({
    "LOGICAL FUNCTION": true,
    "LOGICAL": true,
    "COMPLEX": true
}));
