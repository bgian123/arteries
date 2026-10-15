"""Memory-lean reader for the country road pickles.

The road files produced by the original Overpass download are gzipped pickles
of a Python ``set`` of segments ``((lat1, lon1), (lat2, lon2))``. Unpickling a
large country (USA, CHN: ~400 MB compressed) materialises hundreds of millions
of Python float/tuple objects (tens of GB). Instead, this module interprets the
pickle opcode stream directly with a small Numba-compiled virtual machine that
writes coordinates straight into NumPy arrays (~32 bytes/segment).

Supported opcodes are exactly those emitted by CPython (protocols 2-5) when
pickling a set of 2-tuples of 2-tuples of floats. Any other opcode raises an
error, so the reader can never silently mis-parse a file. ``read_segments``
falls back to ``pickle.load`` when Numba is unavailable, and the test-suite
checks that both paths return identical segment sets.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

try:
    import numba as nb

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

# item kinds on the VM stack
_K_FLOAT, _K_POINT, _K_SEG, _K_MARK, _K_OTHER = 0, 1, 2, 3, 4


def _parse_impl(buf, pt_lat, pt_lon, seg_a, seg_b, memo):
    n = buf.shape[0]
    st_kind = np.empty(1 << 16, dtype=np.int8)
    st_f = np.empty(1 << 16, dtype=np.float64)
    st_i = np.empty(1 << 16, dtype=np.int64)
    sp = 0
    n_pt = 0
    n_seg = 0
    n_memo = 0
    tmp = np.empty(1, dtype=np.uint64)
    tmpf = tmp.view(np.float64)
    pos = 0
    while pos < n:
        op = buf[pos]
        pos += 1
        if op == 0x80:  # PROTO
            pos += 1
        elif op == 0x95:  # FRAME (8-byte length; framing is transparent)
            pos += 8
        elif op == 0x47:  # BINFLOAT, 8 bytes big-endian IEEE754
            v = np.uint64(0)
            for k in range(8):
                v = (v << np.uint64(8)) | np.uint64(buf[pos + k])
            pos += 8
            tmp[0] = v
            st_kind[sp] = _K_FLOAT
            st_f[sp] = tmpf[0]
            sp += 1
        elif op == 0x86:  # TUPLE2
            k1 = st_kind[sp - 2]
            k2 = st_kind[sp - 1]
            if k1 == _K_FLOAT and k2 == _K_FLOAT:
                pt_lat[n_pt] = st_f[sp - 2]
                pt_lon[n_pt] = st_f[sp - 1]
                sp -= 2
                st_kind[sp] = _K_POINT
                st_i[sp] = n_pt
                sp += 1
                n_pt += 1
            elif k1 == _K_POINT and k2 == _K_POINT:
                seg_a[n_seg] = st_i[sp - 2]
                seg_b[n_seg] = st_i[sp - 1]
                sp -= 2
                st_kind[sp] = _K_SEG
                st_i[sp] = n_seg
                sp += 1
                n_seg += 1
            else:
                return -1000 - pos, n_pt, n_seg
        elif op == 0x94:  # MEMOIZE
            if st_kind[sp - 1] == _K_POINT:
                memo[n_memo] = st_i[sp - 1]
            else:
                memo[n_memo] = -1 - st_kind[sp - 1]
            n_memo += 1
        elif op == 0x71 or op == 0x72:  # BINPUT / LONG_BINPUT
            if op == 0x71:
                idx = np.int64(buf[pos])
                pos += 1
            else:
                idx = (np.int64(buf[pos]) | (np.int64(buf[pos + 1]) << 8)
                       | (np.int64(buf[pos + 2]) << 16) | (np.int64(buf[pos + 3]) << 24))
                pos += 4
            if st_kind[sp - 1] == _K_POINT:
                memo[idx] = st_i[sp - 1]
            else:
                memo[idx] = -1 - st_kind[sp - 1]
            if idx + 1 > n_memo:
                n_memo = idx + 1
        elif op == 0x68 or op == 0x6A:  # BINGET / LONG_BINGET
            if op == 0x68:
                idx = np.int64(buf[pos])
                pos += 1
            else:
                idx = (np.int64(buf[pos]) | (np.int64(buf[pos + 1]) << 8)
                       | (np.int64(buf[pos + 2]) << 16) | (np.int64(buf[pos + 3]) << 24))
                pos += 4
            m = memo[idx]
            if m >= 0:
                st_kind[sp] = _K_POINT
                st_i[sp] = m
            else:
                st_kind[sp] = -1 - m
                st_i[sp] = -1
            sp += 1
        elif op == 0x28:  # MARK
            st_kind[sp] = _K_MARK
            sp += 1
        elif op == 0x90:  # ADDITEMS: pop to mark (items already recorded)
            while st_kind[sp - 1] != _K_MARK:
                sp -= 1
            sp -= 1
        elif op == 0x8F or op == 0x7D or op == 0x5D:  # EMPTY_SET / EMPTY_DICT / EMPTY_LIST
            st_kind[sp] = _K_OTHER
            sp += 1
        elif op == 0x2E:  # STOP
            return 0, n_pt, n_seg
        else:
            return -pos, n_pt, n_seg  # unsupported opcode at pos-1
    return -2, n_pt, n_seg


def _count_impl(buf):
    """First pass: count floats, TUPLE2 and memo slots so arrays are sized exactly."""
    n = buf.shape[0]
    n_float = 0
    n_t2 = 0
    n_memo = 0
    pos = 0
    while pos < n:
        op = buf[pos]
        pos += 1
        if op == 0x80 or op == 0x68 or op == 0x71:
            if op == 0x71:
                idx = np.int64(buf[pos])
                if idx + 1 > n_memo:
                    n_memo = idx + 1
            pos += 1
        elif op == 0x95:
            pos += 8
        elif op == 0x47:
            n_float += 1
            pos += 8
        elif op == 0x6A:
            pos += 4
        elif op == 0x72:
            idx = (np.int64(buf[pos]) | (np.int64(buf[pos + 1]) << 8)
                   | (np.int64(buf[pos + 2]) << 16) | (np.int64(buf[pos + 3]) << 24))
            if idx + 1 > n_memo:
                n_memo = idx + 1
            pos += 4
        elif op == 0x86:
            n_t2 += 1
        elif op == 0x94:
            n_memo += 1
        elif op == 0x2E:
            break
    return n_float, n_t2, n_memo


if HAVE_NUMBA:
    _parse = nb.njit(cache=True, nogil=True)(_parse_impl)
    _count = nb.njit(cache=True, nogil=True)(_count_impl)
else:  # pragma: no cover
    _parse = None
    _count = None


def _read_bytes(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    return np.frombuffer(data, dtype=np.uint8)


def read_segments(path, *, force_python: bool = False):
    """Return ``(lat_a, lon_a, lat_b, lon_b)`` float64 arrays for every pickled segment.

    Segments are returned as stored (both directions may be present); de-duplication
    happens later on integer coordinate keys.
    """
    path = Path(path)
    if path.stat().st_size < 64:  # 38-byte files hold an empty set
        try:
            obj = pickle.load(gzip.open(path, "rb"))
            if not obj:
                z = np.zeros(0)
                return z, z.copy(), z.copy(), z.copy()
        except Exception:
            z = np.zeros(0)
            return z, z.copy(), z.copy(), z.copy()

    if HAVE_NUMBA and not force_python:
        buf = _read_bytes(path)
        n_float, n_t2, n_memo = _count(buf)
        n_pt_max = n_float // 2 + 1
        pt_lat = np.empty(n_pt_max, dtype=np.float64)
        pt_lon = np.empty_like(pt_lat)
        seg_max = max(n_t2 - n_float // 2, 0) + 1
        seg_a = np.empty(seg_max, dtype=np.int64)
        seg_b = np.empty(seg_max, dtype=np.int64)
        memo = np.empty(n_memo + 1, dtype=np.int64)
        code, n_pt, n_seg = _parse(buf, pt_lat, pt_lon, seg_a, seg_b, memo)
        del memo, buf
        if code != 0:
            import warnings

            warnings.warn(f"{path.name}: opcode stream not supported by the fast reader "
                          f"(code {code}); falling back to pickle.load")
            return read_segments(path, force_python=True)
        a = seg_a[:n_seg]
        b = seg_b[:n_seg]
        return pt_lat[a], pt_lon[a], pt_lat[b], pt_lon[b]

    with gzip.open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, set):
        raise TypeError(f"{path} does not contain a set")
    arr = np.array([(a[0], a[1], b[0], b[1]) for a, b in obj], dtype=np.float64).reshape(-1, 4)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
