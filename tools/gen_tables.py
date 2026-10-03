#!/usr/bin/env python3
"""Regenerates src/tables.rs from the AV1 specification.

Usage: tools/gen_tables.py <av1-spec source directory> > src/tables.rs

The input is the text of the public "AV1 Bitstream & Decoding Process
Specification" (AOMedia), in the Markdown form AOMedia publishes it from
(https://github.com/AOMediaCodec/av1-spec, the source of
https://aomediacodec.github.io/av1-spec/av1-spec.pdf). Only the
specification's own text is read: every constant lookup table written in it
as `Name[ dims ] = { ... }` (sections 5 to 9 of the specification, files
06.* to 10.*), comments dropped. Symbolic entries (TX_4X4, BLOCK_8X8,
DC_PRED, WEDGE_VERTICAL, SIG_COEF_CONTEXTS_2D + 5, ...) are resolved with
the constants of the specification's section 3 and the enumerations of its
semantics, and every table is checked against its declared dimensions.

The tables are data, reproduced from the specification; nothing else is.
"""
import os
import re
import sys

SPEC_FILES = [
    '06.bitstream.syntax.md',
    '07.bitstream.semantics.md',
    '08.decoding.process.md',
    '09.parsing.process.md',
    '10.additional.tables.md',
]


def constants(spec_dir):
    """Section 3's constants plus the enumerations of the semantics."""
    c = {}
    text = open(os.path.join(spec_dir, '03.symbols.md'), encoding='utf-8').read()
    rows = re.findall(r'^\|\s*`([A-Z_0-9]+)`\s*\|\s*([^|]+?)\s*\|', text, re.M)
    pending = []
    for name, val in rows:
        pending.append((name, val.replace('\\', '')))
    # Values may refer to earlier constants (1 << SUPERRES_FILTER_BITS).
    for _ in range(4):
        for name, val in pending:
            try:
                c[name] = int(eval(val, {}, dict(c)))
            except Exception:
                pass
    enums = {
        'BLOCK': ['4X4', '4X8', '8X4', '8X8', '8X16', '16X8', '16X16', '16X32', '32X16',
                  '32X32', '32X64', '64X32', '64X64', '64X128', '128X64', '128X128',
                  '4X16', '16X4', '8X32', '32X8', '16X64', '64X16'],
        'TX': ['4X4', '8X8', '16X16', '32X32', '64X64', '4X8', '8X4', '8X16', '16X8',
               '16X32', '32X16', '32X64', '64X32', '4X16', '16X4', '8X32', '32X8',
               '16X64', '64X16'],
    }
    for prefix, names in enums.items():
        for i, n in enumerate(names):
            c[f'{prefix}_{n}'] = i
    modes = ['DC_PRED', 'V_PRED', 'H_PRED', 'D45_PRED', 'D135_PRED', 'D113_PRED',
             'D157_PRED', 'D203_PRED', 'D67_PRED', 'SMOOTH_PRED', 'SMOOTH_V_PRED',
             'SMOOTH_H_PRED', 'PAETH_PRED', 'UV_CFL_PRED']
    for i, n in enumerate(modes):
        c[n] = i
    for i, n in enumerate(['WEDGE_HORIZONTAL', 'WEDGE_VERTICAL', 'WEDGE_OBLIQUE27',
                           'WEDGE_OBLIQUE63', 'WEDGE_OBLIQUE117', 'WEDGE_OBLIQUE153']):
        c[n] = i
    # Section 6.10.15: FrameRestorationType.
    c.update(RESTORE_NONE=0, RESTORE_WIENER=1, RESTORE_SGRPROJ=2, RESTORE_SWITCHABLE=3)
    for i, n in enumerate(['INTRA_FRAME', 'LAST_FRAME', 'LAST2_FRAME', 'LAST3_FRAME',
                           'GOLDEN_FRAME', 'BWDREF_FRAME', 'ALTREF2_FRAME', 'ALTREF_FRAME']):
        c[n] = i
    return c


def strip_comments(s):
    s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', s)


def extract(spec_dir):
    """Every `Name[ dims ] = { ... }` inside a code block."""
    out = {}
    for fn in SPEC_FILES:
        text = open(os.path.join(spec_dir, fn), encoding='utf-8').read()
        # Code blocks only: fenced with ~~~~~ (the syntax tables use '|').
        blocks = re.findall(r'~~~~~?\s*c?\n(.*?)~~~~~?', text, flags=re.S)
        for b in blocks:
            b = strip_comments(b)
            for m in re.finditer(r'(?<![A-Za-z0-9_])([A-Z][A-Za-z0-9_]*)\s*((?:\[[^\]]*\]\s*)+)=\s*\{', b):
                i = m.end()
                depth = 1
                while depth:
                    ch = b[i]
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                    i += 1
                body = b[m.end():i - 1]
                dims = re.findall(r'\[([^\]]*)\]', m.group(2))
                # Entries are separated by commas; a line break also ends an entry
                # (Split_Tx_Size in 10.additional.tables.md lacks the comma
                # after its thirteenth entry). The dimension check below
                # catches any other miscount.
                flat = body.replace('{', ',').replace('}', ',').replace('\n', ',')
                elems = [' '.join(e.split()) for e in flat.split(',')]
                out[m.group(1)] = (dims, [e for e in elems if e])
    return out


# name -> (rust name, element type)
CDF = 'u16'
USIZE = [
    'Mi_Width_Log2', 'Mi_Height_Log2', 'Num_4x4_Blocks_Wide', 'Num_4x4_Blocks_High',
    'Size_Group', 'Max_Tx_Size_Rect', 'Max_Tx_Depth', 'Partition_Subsize',
    'Subsampled_Size', 'Split_Tx_Size', 'Tx_Width', 'Tx_Height', 'Tx_Width_Log2',
    'Tx_Height_Log2', 'Tx_Size_Sqr', 'Tx_Size_Sqr_Up', 'Adjusted_Tx_Size', 'Mode_To_Txfm',
    'Wedge_Bits', 'Qm_Offset', 'Intra_Mode_Context', 'Filter_Intra_Mode_To_Intra_Dir',
    'Tx_Type_Intra_Inv_Set1', 'Tx_Type_Intra_Inv_Set2', 'Tx_Type_Inter_Inv_Set1',
    'Tx_Type_Inter_Inv_Set2', 'Tx_Type_Inter_Inv_Set3', 'Ref_Frame_List', 'Remap_Lr_Type',
    'Coeff_Base_Ctx_Offset', 'Coeff_Base_Pos_Ctx_Offset', 'Compound_Mode_Ctx_Map',
    'Cdef_Uv_Dir', 'Segmentation_Feature_Bits', 'Tx_Type_In_Set_Intra',
    'Tx_Type_In_Set_Inter', 'Transform_Row_Shift', 'Wedge_Codebook',
]
U8 = ['Quantizer_Matrix']
U16 = [n for n in []]


def rust_type(name):
    if name.endswith('_Cdf') or name.startswith(('Default_Scan', 'Mrow_Scan', 'Mcol_Scan')):
        return 'u16'
    if name in USIZE:
        return 'usize'
    if name in U8:
        return 'u8'
    return 'i32'


def main():
    spec_dir = sys.argv[1]
    consts = constants(spec_dir)
    tables = extract(spec_dir)
    names = sorted(tables)
    out = []
    for name in names:
        dims_s, elems = tables[name]
        try:
            dims = [int(eval(d, {}, dict(consts))) for d in dims_s]
            vals = [int(eval(e, {}, dict(consts))) for e in elems]
        except NameError as e:
            # Not a constant table (a local array in a process description).
            sys.stderr.write(f'skipped {name}: {e}\n')
            continue
        total = 1
        for d in dims:
            total *= d
        if len(vals) != total:
            raise SystemExit(f'{name}: {len(vals)} values, {dims} declared')
        ty = rust_type(name)

        def build(d, vs):
            if len(d) == 1:
                return '[' + ', '.join(str(v) for v in vs) + ']'
            step = len(vs) // d[0]
            return '[' + ', '.join(build(d[1:], vs[i * step:(i + 1) * step]) for i in range(d[0])) + ']'

        t = ty
        for d in reversed(dims):
            t = f'[{t}; {d}]'
        rust = name.upper()
        decl = '[' + ']['.join(s.strip() for s in dims_s) + ']'
        out.append(f'/// `{name}{decl}`.\npub(crate) static {rust}: {t} = {build(dims, vals)};\n')
    hdr = '''//! Constant tables of the AV1 specification.
//!
//! Transcribed from the AV1 Bitstream & Decoding Process Specification
//! (AOMedia) by `tools/gen_tables.py`, from the text of the specification:
//! default CDFs, scans, quantiser lookups and matrices, filter taps,
//! block-size and transform-size lookups. They are data, reproduced from
//! the specification under its own names (upper-cased). Do not edit by
//! hand: regenerate.

#![allow(dead_code, clippy::unreadable_literal, clippy::large_const_arrays)]

'''
    sys.stdout.write(hdr + '\n'.join(out))


main()
