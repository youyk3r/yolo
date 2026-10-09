"""Minimal OLE2/CFB + Word 97 binary text extractor (no external deps)."""

import struct
import sys


class Cfb:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        hdr = self.data[:512]
        assert hdr[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "not an OLE2 file"
        self.sector_shift = struct.unpack_from("<H", hdr, 30)[0]
        self.mini_shift = struct.unpack_from("<H", hdr, 32)[0]
        self.sector_size = 1 << self.sector_shift
        self.mini_size = 1 << self.mini_shift
        self.num_fat = struct.unpack_from("<I", hdr, 44)[0]
        self.first_dir = struct.unpack_from("<I", hdr, 48)[0]
        self.mini_cutoff = struct.unpack_from("<I", hdr, 56)[0]
        self.first_minifat = struct.unpack_from("<I", hdr, 60)[0]
        self.num_minifat = struct.unpack_from("<I", hdr, 64)[0]
        self.first_difat = struct.unpack_from("<I", hdr, 68)[0]
        self.num_difat = struct.unpack_from("<I", hdr, 72)[0]
        self._read_fat()
        self._read_dir()
        self._read_minifat()

    def _sector(self, idx):
        off = (idx + 1) * self.sector_size
        return self.data[off : off + self.sector_size]

    def _read_fat(self):
        difat = list(struct.unpack_from("<109I", self.data, 76))
        nxt, cnt = self.first_difat, self.num_difat
        while cnt > 0 and nxt not in (0xFFFFFFFE, 0xFFFFFFFF):
            sec = self._sector(nxt)
            difat.extend(struct.unpack_from("<%dI" % (self.sector_size // 4 - 1), sec, 0))
            nxt = struct.unpack_from("<I", sec, self.sector_size - 4)[0]
            cnt -= 1
        self.fat = []
        for s in difat[: self.num_fat]:
            if s >= 0xFFFFFFFE:
                continue
            self.fat.extend(struct.unpack_from("<%dI" % (self.sector_size // 4), self._sector(s), 0))

    def _chain(self, start):
        out, cur, seen = [], start, set()
        while cur < 0xFFFFFFFE and cur not in seen:
            seen.add(cur)
            out.append(cur)
            cur = self.fat[cur] if cur < len(self.fat) else 0xFFFFFFFE
        return out

    def _read_dir(self):
        raw = b"".join(self._sector(s) for s in self._chain(self.first_dir))
        self.entries = []
        for i in range(len(raw) // 128):
            e = raw[i * 128 : (i + 1) * 128]
            nlen = struct.unpack_from("<H", e, 64)[0]
            name = e[: max(0, nlen - 2)].decode("utf-16-le", "ignore")
            etype = e[66]
            start = struct.unpack_from("<I", e, 116)[0]
            size = struct.unpack_from("<Q", e, 120)[0]
            self.entries.append({"name": name, "type": etype, "start": start, "size": size})
        roots = [e for e in self.entries if e["type"] == 5]
        self.ministream = b"".join(self._sector(s) for s in self._chain(roots[0]["start"])) if roots else b""

    def _read_minifat(self):
        raw = b"".join(self._sector(s) for s in self._chain(self.first_minifat))
        self.minifat = list(struct.unpack_from("<%dI" % (len(raw) // 4), raw, 0)) if raw else []

    def stream(self, name):
        for e in self.entries:
            if e["name"] == name and e["type"] == 2:
                if e["size"] < self.mini_cutoff:
                    out, cur, seen = b"", e["start"], set()
                    while cur < 0xFFFFFFFE and cur not in seen:
                        seen.add(cur)
                        out += self.ministream[cur * self.mini_size : (cur + 1) * self.mini_size]
                        cur = self.minifat[cur] if cur < len(self.minifat) else 0xFFFFFFFE
                    return out[: e["size"]]
                return b"".join(self._sector(s) for s in self._chain(e["start"]))[: e["size"]]
        raise KeyError(name)


def extract(path):
    cfb = Cfb(path)
    wd = cfb.stream("WordDocument")
    table_name = None
    for cand in ("1Table", "0Table"):
        try:
            cfb.stream(cand)
            table_name = cand
            break
        except KeyError:
            continue
    table = cfb.stream(table_name)
    csw = struct.unpack_from("<H", wd, 32)[0]
    off = 34 + csw * 2
    cslw = struct.unpack_from("<H", wd, off)[0]
    rg_lw = off + 2
    ccp_text = struct.unpack_from("<I", wd, rg_lw + 3 * 4)[0]
    rg_fc_lcb = rg_lw + cslw * 4 + 2
    fc_clx, lcb_clx = struct.unpack_from("<II", wd, rg_fc_lcb + 33 * 8)
    clx = table[fc_clx : fc_clx + lcb_clx]
    pos, plc = 0, None
    while pos < len(clx):
        kind = clx[pos]
        if kind == 0x01:
            cb = struct.unpack_from("<H", clx, pos + 1)[0]
            pos += 3 + cb
        elif kind == 0x02:
            lcb = struct.unpack_from("<I", clx, pos + 1)[0]
            plc = clx[pos + 5 : pos + 5 + lcb]
            break
        else:
            break
    n = (len(plc) - 4) // 12
    cps = list(struct.unpack_from("<%dI" % (n + 1), plc, 0))
    out = []
    for i in range(n):
        base = 4 * (n + 1) + i * 8
        fc = struct.unpack_from("<I", plc, base + 2)[0]
        compressed = bool(fc & 0x40000000)
        fc = fc & 0x3FFFFFFF
        ncp = cps[i + 1] - cps[i]
        if compressed:
            raw = wd[fc // 2 : fc // 2 + ncp]
            out.append(raw.decode("cp1252", "replace"))
        else:
            raw = wd[fc : fc + ncp * 2]
            out.append(raw.decode("utf-16-le", "replace"))
    return "".join(out)[:ccp_text]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(extract(sys.argv[1]))
