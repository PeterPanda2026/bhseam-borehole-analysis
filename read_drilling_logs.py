"""
Read the soil/rock layers of each borehole from the drilling-log workbook
supplied with this project (drilling_logs.xlsx).

The workbook is the paper drilling-log form repeated once per borehole, one
block after another, exactly as the site staff typed it in. Columns (0-based):
    0  line letter   (index column) 5  depth FROM ('gl' = ground level)
    1  column number (index column) 6  depth TO BASE OF STRATA
    2  line number within the form  7  casing depth
    3  description of strata        8  bit size (mm)
    4  (rig / title text)           9  comments  (also the 'BH No.' value)
                                   10  plot number
Columns 0 and 1 are an index added when the records were typed up: together
they repeat the hole number on every row of a block (0 + 1 = the BH No. in
column 9), so they hold nothing new, but they make a sheet easy to filter.
Blocks are found by their labels ("DRILLING LOG", "DESCRIPTION OF STRATA",
"DELAYS/STANDING TIME") rather than by fixed row offsets, because a few blocks
are shifted by a row.

What it gives you: one row per layer, with the borehole header (hole number,
plot, driller, rig) repeated on each of its layers, written to
strata_layers.csv.

Hole positions are in borehole_locations.csv; join the two on the hole number.

What it does NOT do, and you have to work out yourself: which coal seam each
coal layer belongs to, and the depth and shape of each seam along the section.

Requires: pandas, openpyxl   ->   pip install pandas openpyxl
"""

from pathlib import Path

import pandas as pd

XLS_PATH = Path(__file__).with_name("drilling_logs.xlsx")

COL_DESC, COL_FROM, COL_TO, COL_CASING, COL_BIT, COL_COMMENT, COL_PLOTNO = 3, 5, 6, 7, 8, 9, 10


def _txt(v) -> str:
    """Cell value as a stripped string ('' for blanks/NaN)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _depth(v):
    """Parse a depth cell: 'gl' -> 0.0, number -> float, blank -> None."""
    s = _txt(v).lower()
    if not s:
        return None
    if s in ("gl", "g.l.", "g.l"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _find(block: pd.DataFrame, label: str, col: int):
    """Index (within block) of the first row whose `col` starts with `label`."""
    for i, v in enumerate(block[col]):
        if _txt(v).upper().startswith(label):
            return i
    return None


def split_blocks(df: pd.DataFrame):
    """Yield one DataFrame per borehole form, ending at 'DELAYS/STANDING TIME'."""
    start = 0
    for i, v in enumerate(df[COL_DESC]):
        if _txt(v).upper().startswith("DELAYS"):
            yield df.iloc[start : i + 1].reset_index(drop=True)
            start = i + 1


def parse_block(block: pd.DataFrame):
    """Return (header dict, list of layer dicts) for one borehole form."""
    hdr = {"plot": _txt(block.iat[0, 0])}

    # Borehole ID and plot number sit on the 'DRILLING LOG' row, cols 9/10.
    r = _find(block, "DRILLING LOG", 5)
    if r is not None:
        hdr["borehole"] = _txt(block.iat[r, COL_COMMENT]).replace(" ", "")
        hdr["plot_no"] = _txt(block.iat[r, COL_PLOTNO])
    else:  # fall back to the index columns
        hdr["borehole"] = f"{hdr['plot']}{_txt(block.iat[0, 1])}"

    # Driller / rig are one row below their labels; date drilled likewise.
    r = _find(block, "DRILLER", COL_DESC)
    if r is not None:
        hdr["driller"] = _txt(block.iat[r + 1, 3])
        hdr["rig"] = _txt(block.iat[r + 1, 4])
        hdr["ground_level"] = _txt(block.iat[r + 1, COL_COMMENT])  # blank in this file
    r = _find(block, "INCLINATION", COL_DESC)
    if r is not None:
        hdr["inclination"] = _txt(block.iat[r + 1, 3])
        hdr["flush"] = _txt(block.iat[r + 1, 4])
        hdr["date_drilled"] = _txt(block.iat[r + 1, COL_COMMENT])

    # Strata table: rows between the 'DESCRIPTION OF STRATA' header and the
    # 'DELAYS' footer. The header spans 3 lines ('TO BASE' / 'OF' / 'STRATA').
    top = _find(block, "DESCRIPTION OF STRATA", COL_DESC)
    bottom = len(block) - 1
    layers, pending_comments = [], []
    if top is None:
        return hdr, layers

    for i in range(top + 1, bottom):
        row = block.iloc[i]
        desc = _txt(row[COL_DESC])
        d_from, d_to = _depth(row[COL_FROM]), _depth(row[COL_TO])
        comment = _txt(row[COL_COMMENT])
        if _txt(row[COL_TO]).upper() in ("OF", "STRATA"):  # header continuation
            continue

        if desc or d_to is not None:
            if pending_comments:  # comment lines written above this layer
                comment = "; ".join(pending_comments + ([comment] if comment else []))
                pending_comments = []
            if d_from is None and layers:  # FROM left blank -> previous base
                d_from = layers[-1]["depth_to"]
            layers.append(
                {
                    "description": desc,
                    "depth_from": d_from,
                    "depth_to": d_to,
                    "thickness": (d_to - d_from) if None not in (d_from, d_to) else None,
                    "casing": _depth(row[COL_CASING]),
                    "bit_size": _txt(row[COL_BIT]),
                    "comments": comment,
                }
            )
        elif comment:  # comment-only line (e.g. 'FWR' above the first layer)
            pending_comments.append(comment)

    if pending_comments and layers:  # trailing comments -> last layer
        extra = "; ".join(pending_comments)
        layers[-1]["comments"] = "; ".join(filter(None, [layers[-1]["comments"], extra]))
    return hdr, layers


def read_drill_logs(path=XLS_PATH):
    """
    Returns
    -------
    layers : DataFrame, one row per stratum, with borehole header columns.
    boreholes : dict  {borehole_id: DataFrame of that borehole's layers}
    """
    records = []
    # every worksheet (one per plot)
    for df in pd.read_excel(path, sheet_name=None, header=None, dtype=object).values():
        if df.empty:
            continue
        df = df.reindex(columns=range(max(11, df.shape[1])))  # narrow sheets -> pad with blanks
        for block in split_blocks(df):
            hdr, layers = parse_block(block)
            for n, layer in enumerate(layers, start=1):
                records.append({**hdr, "layer_no": n, **layer})

    layers = pd.DataFrame(records)
    if layers.empty:
        return layers, {}
    boreholes = {bh: g.reset_index(drop=True) for bh, g in layers.groupby("borehole", sort=False)}
    return layers, boreholes


if __name__ == "__main__":
    layers, boreholes = read_drill_logs()
    print(f"{len(boreholes)} boreholes, {len(layers)} layers\n")

    for bh in list(boreholes)[:3]:  # preview a few
        print(f"=== {bh}")
        print(boreholes[bh][["layer_no", "description", "depth_from", "depth_to", "comments"]]
              .to_string(index=False), "\n")

    out = Path(__file__).with_name("strata_layers.csv")
    layers.to_csv(out, index=False)
    print(f"Saved -> {out}")
