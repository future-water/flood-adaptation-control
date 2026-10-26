"""
Build COMID -> HUC8/HUC10/HUC12 mapping CSV.

Inputs (relative to this script):
  shape/Flowlines_huc02_12_Brazos.shp   NHDPlus flowlines clipped to the Brazos
                                        (COMID, TotDASqKM, StreamOrde, REACHCODE,
                                        geometry)
  shape/WBDHU12_Brazos.shp              USGS WBD HUC12 polygons clipped to the
                                        Brazos (huc12, geometry)

Output:
  comid_huc_mapping_correct.csv         COMID, TotDASqKM, StreamOrde,
                                        HUC8, HUC10, HUC12

Two derivations are computed and compared:
  (A) REACHCODE-based:  HUC12 = REACHCODE[:12]  (NHDPlus self-attribution)
  (B) Spatial-join:     HUC12 = WBDHU12 containing the flowline midpoint

The script writes (B) by default. Use --method reachcode to write (A) instead.

Usage:
    python build_comid_huc_mapping.py
    python build_comid_huc_mapping.py --method reachcode
    python build_comid_huc_mapping.py --compare  # write both and diff
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


FLOWLINES = Path("shape/Flowlines_huc02_12_Brazos.shp")
# WBDHU12 is searched in this order. With the bundled Brazos-clipped layer,
# 3 flowlines at the basin edge fall outside the polygon coverage (their
# representative points sit in adjacent HUC8s — Galveston Bay, San Jacinto,
# Guadalupe). They are dropped from the output; downstream analysis filters
# them out anyway because they have no streamflow data.
WBDHU12_CANDIDATES = [
    Path("shape/WBDHU12.shp"),
    Path("shape/WBDHU12_Brazos.shp"),
]
OUT_CSV = Path("comid_huc_mapping_correct.csv")


def load_flowlines() -> gpd.GeoDataFrame:
    fl = gpd.read_file(FLOWLINES)[
        ["COMID", "TotDASqKM", "StreamOrde", "REACHCODE", "geometry"]
    ].copy()
    fl["COMID"] = fl["COMID"].astype("int64")
    fl["StreamOrde"] = fl["StreamOrde"].astype("int64")
    fl["TotDASqKM"] = fl["TotDASqKM"].astype("float64")
    return fl


def load_wbd_huc12(path: Path | None = None) -> gpd.GeoDataFrame:
    if path is None:
        path = next((p for p in WBDHU12_CANDIDATES if p.exists()), None)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"No WBDHU12 shapefile found. Looked for: {WBDHU12_CANDIDATES}"
        )
    print(f"      WBD source: {path}")
    w = gpd.read_file(path)
    huc_col = next(c for c in w.columns if c.lower() == "huc12")
    w = w[[huc_col, "geometry"]].rename(columns={huc_col: "HUC12"})
    w["HUC12"] = w["HUC12"].astype(str).str.strip()
    return w


def derive_from_reachcode(fl: gpd.GeoDataFrame) -> pd.DataFrame:
    """HUC12 = REACHCODE[:12], then slice HUC10/HUC8."""
    df = fl.drop(columns="geometry").copy()
    df["HUC12"] = df["REACHCODE"].astype(str).str[:12]
    df["HUC10"] = df["HUC12"].str[:10]
    df["HUC8"]  = df["HUC12"].str[:8]
    return df.drop(columns="REACHCODE")


def derive_from_sjoin(fl: gpd.GeoDataFrame, wbd12: gpd.GeoDataFrame) -> pd.DataFrame:
    """HUC12 = WBDHU12 polygon containing each flowline's representative point.

    `representative_point()` is preferred over `interpolate(0.5)` because it is
    guaranteed to lie within the geometry, which avoids ambiguous boundary
    placements at HUC seams. A projected equal-area CRS (EPSG:5070) is used so
    that the geometric operations are metric rather than degree-based.
    """
    fl_p = fl.to_crs(5070)
    wbd12_p = wbd12.to_crs(5070)

    rep = fl_p.geometry.representative_point()
    pts = gpd.GeoDataFrame(
        {"COMID": fl_p["COMID"].values},
        geometry=rep,
        crs=fl_p.crs,
    )

    joined = gpd.sjoin(pts, wbd12_p, how="left", predicate="within")[["COMID", "HUC12"]]
    joined = joined.drop_duplicates(subset="COMID", keep="first")

    df = fl.drop(columns=["geometry", "REACHCODE"]).merge(joined, on="COMID", how="left")
    df["HUC10"] = df["HUC12"].str[:10]
    df["HUC8"]  = df["HUC12"].str[:8]
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Match dtypes and column order of the existing CSV."""
    out = df[["COMID", "TotDASqKM", "StreamOrde", "HUC8", "HUC10", "HUC12"]].copy()
    n_missing = out[["HUC8", "HUC10", "HUC12"]].isna().any(axis=1).sum()
    if n_missing:
        print(f"  WARNING: {n_missing} rows have missing HUC labels; dropped.")
        out = out.dropna(subset=["HUC8", "HUC10", "HUC12"])
    for col in ["HUC8", "HUC10", "HUC12"]:
        out[col] = out[col].astype("int64")
    out["COMID"] = out["COMID"].astype("int64")
    out["StreamOrde"] = out["StreamOrde"].astype("int64")
    return out.reset_index(drop=True)


def compare(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> None:
    common = sorted(set(a["COMID"]).intersection(b["COMID"]))
    a_idx = a.set_index("COMID").loc[common]
    b_idx = b.set_index("COMID").loc[common]
    print(f"\n  {label_a} vs {label_b}: {len(common)} common COMIDs")
    for col in ["HUC8", "HUC10", "HUC12"]:
        ndiff = (a_idx[col] != b_idx[col]).sum()
        print(f"    {col}: {ndiff} differ ({ndiff/len(common):.2%})")


def verify_against(generated: pd.DataFrame, reference_csv: str) -> None:
    """Report how closely `generated` matches a reference CSV, both row-wise
    (HUC labels) and downstream (per-HUC outlet selection)."""
    ref = pd.read_csv(reference_csv)
    print(f"\n  Reference: {reference_csv} ({len(ref)} rows)")
    print(f"  Generated: {len(generated)} rows")
    print(f"  COMID set equal: {set(generated.COMID) == set(ref.COMID)}")

    m = generated.merge(ref, on="COMID", suffixes=("_gen", "_ref"))
    print(f"  Matched on COMID: {len(m)}")
    for col in ["TotDASqKM", "StreamOrde", "HUC8", "HUC10", "HUC12"]:
        a, b = m[f"{col}_gen"], m[f"{col}_ref"]
        if pd.api.types.is_float_dtype(a):
            eq = (abs(a - b) < 1e-9) | (a.isna() & b.isna())
        else:
            eq = (a == b)
        ndiff = (~eq).sum()
        tag = "EXACT MATCH" if ndiff == 0 else f"({ndiff/len(m):.4%})"
        print(f"    {col}: {ndiff} differ  {tag}")

    print("\n  Outlet-per-HUC equivalence (max-TotDASqKM reach):")
    for level in ["HUC8", "HUC10", "HUC12"]:
        out_gen = generated.loc[generated.groupby(level)["TotDASqKM"].idxmax()].set_index(level)["COMID"]
        out_ref = ref.loc[ref.groupby(level)["TotDASqKM"].idxmax()].set_index(level)["COMID"]
        common = out_gen.index.intersection(out_ref.index)
        same = (out_gen.loc[common] == out_ref.loc[common]).sum()
        print(f"    {level}: {same}/{len(common)} outlets agree "
              f"(units gen={len(out_gen)}, ref={len(out_ref)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sjoin", "reachcode"], default="sjoin")
    parser.add_argument("--compare", action="store_true",
                        help="Compute both methods and report differences")
    parser.add_argument("--wbd", default=None,
                        help="Path to a WBDHU12 shapefile (defaults to "
                             "shape/WBDHU12.shp, then shape/WBDHU12_Brazos.shp)")
    parser.add_argument("--out", default=str(OUT_CSV))
    parser.add_argument("--verify", default=None,
                        help="Path to a reference CSV; report row-wise and "
                             "outlet-wise differences against the generated mapping.")
    args = parser.parse_args()

    print("[1/3] Loading flowlines ...")
    fl = load_flowlines()
    print(f"      {len(fl)} flowlines")

    if args.method == "sjoin" or args.compare:
        print("[2/3] Loading WBDHU12 polygons ...")
        wbd12 = load_wbd_huc12(Path(args.wbd) if args.wbd else None)
        print(f"      {len(wbd12)} HUC12 polygons")
        df_sjoin = finalize(derive_from_sjoin(fl, wbd12))
    else:
        df_sjoin = None

    if args.method == "reachcode" or args.compare:
        df_rc = finalize(derive_from_reachcode(fl))
    else:
        df_rc = None

    if args.compare and df_sjoin is not None and df_rc is not None:
        compare(df_sjoin, df_rc, "sjoin", "reachcode")

    chosen = df_sjoin if args.method == "sjoin" else df_rc
    print(f"[3/3] Writing {args.out} ({len(chosen)} rows)")
    chosen.to_csv(args.out, index=False)

    if args.verify:
        print(f"\n[verify] Comparing against {args.verify} ...")
        verify_against(chosen, args.verify)


if __name__ == "__main__":
    main()
