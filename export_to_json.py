"""
export_to_json.py  --  Run once (with UU VPN active) to export the DB to
static JSON files that the portal can load without any live DB connection.

Output files (put these next to portal.html):
    data/wells.json          -- all well metadata, one object per well
    data/timeseries/         -- one JSON file per well ID, monthly GWH series

Run with:  python export_to_json.py
"""

import json
import math
import os
import zipfile
import io
import psycopg2
import psycopg2.extras
import pandas as pd
import geopandas as gpd
import requests

# Always write data/ next to this script, regardless of where it is run from
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Working directory: {os.getcwd()}")

# ---------------------------------------------------------------------------
DB = dict(
    dbname="geowat",
    user="geowat_user",
    host="ages-db01.geo.uu.nl",
    password="utrecht1994",
    port=5432,
)

YEAR_START = 1979
YEAR_END   = 2019

OUT_DIR   = "data"
TS_DIR    = os.path.join(OUT_DIR, "timeseries")
LITHO_DIR = os.path.join(OUT_DIR, "lithology")
GWS_DIR   = os.path.join(OUT_DIR, "salinity")
GWE_DIR   = os.path.join(OUT_DIR, "extraction")
os.makedirs(TS_DIR,   exist_ok=True)
os.makedirs(LITHO_DIR, exist_ok=True)
os.makedirs(GWS_DIR,  exist_ok=True)
os.makedirs(GWE_DIR,  exist_ok=True)

# ---------------------------------------------------------------------------

def clean(val):
    """Convert NaN / inf to None so json.dumps produces valid JSON."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def clean_row(row):
    return {k: clean(v) for k, v in row.items()}


def linear_trend(values):
    """Return slope (m per observation step) via least squares. Returns None if <2 points."""
    n = len(values)
    if n < 2:
        return None
    sumx = sumx2 = sumy = sumxy = 0
    for i, y in enumerate(values):
        sumx  += i
        sumx2 += i * i
        sumy  += y
        sumxy += i * y
    denom = n * sumx2 - sumx * sumx
    if denom == 0:
        return None
    return round((n * sumxy - sumx * sumy) / denom, 6)


def month_sum_expr():
    return " + ".join(
        f'CASE WHEN g."{m:02d}_gw_head_m" IS NOT NULL THEN 1 ELSE 0 END'
        for m in range(1, 13)
    )


def month_select_expr():
    return ", ".join(
        f'g."{m:02d}_gw_head_m" / 100.0 AS "{m:02d}_gw_head_m"'
        for m in range(1, 13)
    )


def gws_month_sum_expr():
    return " + ".join(
        f'CASE WHEN g."{m:02d}_gw_salinity_ppm" IS NOT NULL THEN 1 ELSE 0 END'
        for m in range(1, 13)
    )


def gwe_month_sum_expr():
    return " + ".join(
        f'CASE WHEN g."{m:02d}_gw_extraction_m3d" IS NOT NULL THEN 1 ELSE 0 END'
        for m in range(1, 13)
    )


def generic_ts_export(conn, table, col_pattern, out_dir, label, year_start, year_end):
    """
    Generic time series exporter for salinity and extraction tables.
    col_pattern: e.g. '{m:02d}_gw_salinity_ppm'
    Writes one JSON per well: {wellID, n, series:[{date, value}]}
    Also returns {wellID: annual_mean} dict.
    """
    cols = [col_pattern.format(m=m) for m in range(1, 13)]
    col_sql = ", ".join(f'g."{c}"' for c in cols)
    sql = f"""
        SELECT g.id_gerbil, g.year, {col_sql}
        FROM public.{table} g
        WHERE g.year BETWEEN {year_start} AND {year_end}
        ORDER BY g.id_gerbil, g.year
    """
    current_well = None
    current_series = []
    written = 0
    means = {}  # wellID -> overall mean value

    def flush(well_id, series):
        path = os.path.join(out_dir, f"{well_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"wellID": well_id, "n": len(series), "series": series},
                      f, separators=(",", ":"))
        vals = [d["value"] for d in series if d["value"] is not None]
        means[well_id] = round(sum(vals) / len(vals), 4) if vals else None

    with conn.cursor(name=f"{table}_export", cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.itersize = 2000
        cur.execute(sql)
        for row in cur:
            well_id = row["id_gerbil"]
            year    = row["year"]
            if well_id != current_well:
                if current_well is not None:
                    flush(current_well, current_series)
                    written += 1
                    if written % 1000 == 0:
                        print(f"  {label}: {written} wells done...")
                current_well   = well_id
                current_series = []
            for month in range(1, 13):
                val = row[cols[month - 1]]
                if val is not None:
                    v = clean(float(val))
                    if v is not None:
                        current_series.append({
                            "date": f"{year}-{month:02d}",
                            "value": round(v, 4),
                        })
        if current_well is not None:
            flush(current_well, current_series)
            written += 1

    print(f"  {label}: wrote {written} files to {out_dir}/")
    return means


# ---------------------------------------------------------------------------
# WELLS METADATA
# ---------------------------------------------------------------------------
def export_wells(conn):
    print("Exporting well metadata...")

    sql = f"""
        WITH obs AS (
            SELECT
                g.id_gerbil,
                SUM({month_sum_expr()}) AS num_months
            FROM public._gwh_monthly_tb g
            WHERE g.year BETWEEN {YEAR_START} AND {YEAR_END}
            GROUP BY g.id_gerbil
        )
        SELECT DISTINCT ON (l.id_gerbil)
            l.id_gerbil      AS "wellID",
            l.x_wgs84        AS lon,
            l.y_wgs84        AS lat,
            l.country_name,
            l.state_name,
            l.litho,
            l.gwh,
            l.gws,
            l.gwe,
            l.orig_elev_m_asl,
            l.glo90_elev_m_asl,
            COALESCE(obs.num_months, 0) AS num_months
        FROM public._lookup_tb l
        LEFT JOIN obs ON obs.id_gerbil = l.id_gerbil
        ORDER BY l.id_gerbil
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    wells = [clean_row(dict(r)) for r in rows]

    out_path = os.path.join(OUT_DIR, "wells.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(wells), "wells": wells}, f, separators=(",", ":"))

    print(f"  Wrote {len(wells)} wells to {out_path}")
    return wells


def embed_trends(wells, trends):
    """Merge trend slopes into wells list and rewrite wells.json."""
    print("Embedding trend slopes into wells.json...")
    for w in wells:
        w["trend_slope"] = trends.get(w["wellID"])   # m/month, None if no data
    out_path = os.path.join(OUT_DIR, "wells.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(wells), "wells": wells}, f, separators=(",", ":"))
    print(f"  Done. trend_slope embedded for {sum(1 for w in wells if w['trend_slope'] is not None)} wells.")


# ---------------------------------------------------------------------------
# TIME SERIES  (one file per well)
# ---------------------------------------------------------------------------
def export_timeseries(conn, well_ids):
    print(f"Exporting time series for {len(well_ids)} wells...")

    sql = f"""
        SELECT
            g.id_gerbil,
            g.year,
            {month_select_expr()}
        FROM public._gwh_monthly_tb g
        WHERE g.year BETWEEN {YEAR_START} AND {YEAR_END}
        ORDER BY g.id_gerbil, g.year
    """

    current_well = None
    current_series = []
    written = 0
    trends = {}   # wellID -> slope in m/month (None if insufficient data)

    def flush(well_id, series):
        path = os.path.join(TS_DIR, f"{well_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"wellID": well_id, "n": len(series), "series": series},
                      f, separators=(",", ":"))
        # compute trend from the gwh_m values
        vals = [d["gwh_m"] for d in series if d["gwh_m"] is not None]
        trends[well_id] = linear_trend(vals)

    with conn.cursor(name="ts_export", cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.itersize = 2000
        cur.execute(sql)

        for row in cur:
            well_id = row["id_gerbil"]
            year    = row["year"]

            if well_id != current_well:
                if current_well is not None:
                    flush(current_well, current_series)
                    written += 1
                    if written % 500 == 0:
                        print(f"  {written} wells done...")
                current_well   = well_id
                current_series = []

            for month in range(1, 13):
                val = row[f"{month:02d}_gw_head_m"]
                if val is not None:
                    v = clean(float(val))
                    if v is not None:
                        current_series.append({
                            "date": f"{year}-{month:02d}",
                            "gwh_m": round(v, 4),
                        })

        if current_well is not None:
            flush(current_well, current_series)
            written += 1

    print(f"  Done. Wrote {written} time series files to {TS_DIR}/")
    return trends



# ---------------------------------------------------------------------------
# LITHOLOGY  (one file per well from _bore_litho_tb + litho_classes_tb lookup)
# ---------------------------------------------------------------------------
# Depth interval columns in _bore_litho_tb in order
LITHO_DEPTH_COLS = [
    "0m_5m","5m_10m","10m_15m","15m_20m","20m_25m","25m_30m","30m_35m",
    "35m_40m","40m_45m","45m_50m","50m_55m","55m_60m","60m_65m","65m_70m",
    "70m_75m","75m_80m","80m_85m","85m_90m","90m_95m","95m_100m",
    "100m_110m","110m_120m","120m_130m","130m_140m","140m_150m",
    "150m_160m","160m_170m","170m_180m","180m_190m","190m_200m",
    "200m_225m","225m_250m","250m_275m","275m_300m","300m_325m",
    "325m_350m","350m_375m","375m_400m","400m_425m","425m_450m",
    "450m_475m","475m_500m","500m_550m","550m_600m","600m_650m",
    "650m_700m","700m_750m","750m_800m","800m_850m","850m_900m",
    "900m_950m","950m_1000m","1000m_1100m","1100m_1200m","1200m_1300m",
    "1300m_1400m","1400m_1500m","1500m_1600m","1600m_1700m","1700m_1800m",
    "1800m_1900m","1900m_2000m","2000m_2500m","2500m_3000m","3000m_3500m",
    "3500m_4000m","4000m_4500m","4500m_5000m"
]

def export_lithology(conn):
    """
    Writes two files instead of 110k individual files:
      data/lithology/litho_classes.json  -- class id -> name/shortcut
      data/lithology/lithology_all.json  -- {wellID: [[depth_col, litho_id], ...]}

    The portal loads lithology_all.json once at boot into a JS dict,
    so per-well lookup is instant with no additional fetches.
    """
    print("Exporting lithology classes lookup...")
    sql_classes = "SELECT id_litho, glim_name, glim_shortcut FROM public.litho_classes_tb ORDER BY id_litho"
    df_classes = pd.read_sql(sql_classes, conn)
    classes = df_classes.to_dict(orient="records")
    with open(os.path.join(LITHO_DIR, "litho_classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, separators=(",", ":"))
    print(f"  Wrote {len(classes)} lithology classes.")

    print("Exporting bore lithology profiles into single combined file...")
    col_list = ", ".join(f'"{c}"' for c in LITHO_DEPTH_COLS)
    sql = f"""
        SELECT id_gerbil, {col_list}
        FROM public._bore_litho_tb
        WHERE id_gerbil IS NOT NULL
        ORDER BY id_gerbil
    """
    # Store as {wellID: [[depth_col, litho_id], ...]} — compact array format
    all_profiles = {}
    read = 0
    with conn.cursor(name="litho_export", cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.itersize = 5000
        cur.execute(sql)
        for row in cur:
            well_id = row["id_gerbil"]
            profile = [
                [col, int(row[col])]
                for col in LITHO_DEPTH_COLS
                if row[col] is not None
            ]
            if profile:
                all_profiles[well_id] = profile
            read += 1
            if read % 10000 == 0:
                print(f"  {read} rows read...")

    out_path = os.path.join(LITHO_DIR, "lithology_all.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Done. {len(all_profiles)} profiles written to {out_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# REGION ASSIGNMENT  (spatial join against Natural Earth Admin-1)
# ---------------------------------------------------------------------------
NE_ADMIN1_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_1_states_provinces.zip"
)
NE_CACHE = os.path.join(OUT_DIR, "_ne_admin1.gpkg")   # cached locally

def load_admin1():
    """Load Natural Earth Admin-1 shapefile, caching locally as GeoPackage."""
    if os.path.exists(NE_CACHE):
        print(f"  Loading cached admin-1 from {NE_CACHE}")
        return gpd.read_file(NE_CACHE)

    print(f"  Downloading Natural Earth Admin-1 from naciscdn.org ...")
    r = requests.get(NE_ADMIN1_URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # Find the .shp inside the zip
    shp_name = next(n for n in z.namelist() if n.endswith(".shp"))
    with z as zf:
        tmpdir = os.path.join(OUT_DIR, "_ne_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        zf.extractall(tmpdir)
    gdf = gpd.read_file(os.path.join(tmpdir, shp_name))
    gdf.to_file(NE_CACHE, driver="GPKG")
    print(f"  Cached to {NE_CACHE}")
    return gdf


def assign_regions(wells):
    """
    Spatial join well coords against Natural Earth Admin-1 polygons.
    Adds 'region_name' field to each well dict (province/state/territory name).
    Falls back to country_name if no polygon matches.
    """
    print("Assigning regions via Natural Earth Admin-1 spatial join...")
    try:
        admin1 = load_admin1()
    except Exception as e:
        print(f"  WARNING: Could not load admin-1 shapefile: {e}")
        print("  Skipping region assignment — region_name will be empty.")
        for w in wells:
            w["region_name"] = None
        return

    # Build GeoDataFrame of wells with valid coords
    valid = [(i, w) for i, w in enumerate(wells) if w.get("lon") and w.get("lat")]
    if not valid:
        for w in wells:
            w["region_name"] = None
        return

    import numpy as np
    from shapely.geometry import Point

    print(f"  Joining {len(valid)} wells against {len(admin1)} admin-1 polygons...")
    # Keep only needed columns from admin1
    admin1 = admin1[["name", "geometry"]].copy()
    admin1 = admin1.to_crs("EPSG:4326")

    # Process in batches of 50k to avoid memory issues
    BATCH = 50_000
    region_map = {}   # well index -> region_name

    for start in range(0, len(valid), BATCH):
        batch = valid[start:start + BATCH]
        indices = [i for i, _ in batch]
        pts = gpd.GeoDataFrame(
            {"idx": indices},
            geometry=[Point(w["lon"], w["lat"]) for _, w in batch],
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts, admin1, how="left", predicate="within")
        for _, row in joined.iterrows():
            region_map[int(row["idx"])] = row.get("name") if pd.notna(row.get("name")) else None
        print(f"    {min(start + BATCH, len(valid))}/{len(valid)} wells processed...")

    for i, w in enumerate(wells):
        w["region_name"] = region_map.get(i)

    assigned = sum(1 for w in wells if w["region_name"])
    print(f"  Done. {assigned}/{len(wells)} wells assigned a region name.")

# ---------------------------------------------------------------------------
# COUNTRIES LIST  (for filter dropdown)
# ---------------------------------------------------------------------------
def export_countries(conn):
    print("Exporting country list...")
    sql = """
        SELECT DISTINCT country_name
        FROM public._lookup_tb
        WHERE country_name IS NOT NULL
        ORDER BY country_name
    """
    df = pd.read_sql(sql, conn)
    countries = df["country_name"].tolist()
    out_path = os.path.join(OUT_DIR, "countries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"countries": countries}, f, separators=(",", ":"))
    print(f"  Wrote {len(countries)} countries to {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Connecting to {DB['host']}...")
    conn = psycopg2.connect(**DB)
    print("Connected.\n")

    export_countries(conn)
    export_lithology(conn)
    wells = export_wells(conn)
    assign_regions(wells)
    trends = export_timeseries(conn, [w["wellID"] for w in wells])
    embed_trends(wells, trends)

    print("\nExporting salinity time series...")
    gws_means = generic_ts_export(
        conn, "_gws_monthly_tb", "{m:02d}_gw_salinity_ppm",
        GWS_DIR, "Salinity", YEAR_START, YEAR_END
    )
    print("\nExporting extraction time series...")
    gwe_means = generic_ts_export(
        conn, "_gwe_monthly_tb", "{m:02d}_gw_extraction_m3d",
        GWE_DIR, "Extraction", YEAR_START, YEAR_END
    )

    # Embed means into wells.json so the portal can show them without fetching
    print("\nEmbedding salinity and extraction means into wells.json...")
    for w in wells:
        w["gws_mean_ppm"] = gws_means.get(w["wellID"])
        w["gwe_mean_m3d"] = gwe_means.get(w["wellID"])
    out_path = os.path.join(OUT_DIR, "wells.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(wells), "wells": wells}, f, separators=(",", ":"))
    print("  Done.")

    conn.close()
    print("\nExport complete. Copy the data/ folder next to portal.html.")