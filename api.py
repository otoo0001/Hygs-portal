"""
api.py  --  Flask backend for the HYGS groundwater portal.
Run with:  python api.py
Requires:  pip install flask flask-cors psycopg2-binary pandas geopandas

Must be run on a machine with access to ages-db01.geo.uu.nl (UU VPN or campus).

Database tables used:
    public._lookup_tb       -- well metadata (coords, country, boolean flags, elevation)
    public._gwh_monthly_tb  -- monthly groundwater head observations (01_gw_head_m .. 12_gw_head_m)
    public.litho_classes_tb -- standalone lithology reference (no FK to _lookup_tb)

    Note: litho, gwh, gws, gwe in _lookup_tb are all boolean flags, not foreign keys.
    litho_classes_tb has no join path to _lookup_tb and is not used in queries.

Design principles:
    - /api/wells            aggregation in SQL, one row per well returned
    - /api/wells/countries  distinct country list for filter dropdown
    - /api/timeseries       only the requested well's rows fetched
    - /api/export/csv       server-side cursor, never loads full table into RAM
    - /api/export/shp       one row per well only (locations + metadata)
    - /api/health           DB connectivity check
"""

import io
import os
import zipfile
import tempfile
import logging

import pandas as pd
import geopandas as gpd
import psycopg2
import psycopg2.extras

from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS

# ---------------------------------------------------------------------------
# DB CONFIG
# ---------------------------------------------------------------------------
DB = dict(
    dbname="geowat",
    user="geowat_user",
    host="ages-db01.geo.uu.nl",
    password="utrecht1994",
    port=5432,
)

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL BUILDING BLOCKS
# ---------------------------------------------------------------------------

def _month_sum_expr():
    """
    Counts non-null monthly head columns per yearly row in SQL.
    Summed over years in the obs CTE to give total months per well.
    """
    return " + ".join(
        f'CASE WHEN g."{m:02d}_gw_head_m" IS NOT NULL THEN 1 ELSE 0 END'
        for m in range(1, 13)
    )


def _month_select_expr():
    """
    SELECT fragment for all 12 monthly columns, converting cm to m in SQL.
    """
    return ", ".join(
        f'g."{m:02d}_gw_head_m" / 100.0 AS "{m:02d}_gw_head_m"'
        for m in range(1, 13)
    )


def _build_filter_clauses(params):
    """
    Translate all portal filter params into a WHERE clause fragment and a
    psycopg2 named parameter dict.

    Supported params:
        year_start, year_end      -- applied to _gwh_monthly_tb in the obs CTE
        min_months                -- filters on obs.num_months after aggregation
        lon_min, lon_max          -- bounding box on x_wgs84
        lat_min, lat_max          -- bounding box on y_wgs84
        country                   -- exact match on l.country_name
        litho_only                -- if "true", only wells where l.litho IS TRUE
        gwh_only                  -- if "true", only wells where l.gwh IS TRUE
        gws_only                  -- if "true", only wells where l.gws IS TRUE
        gwe_only                  -- if "true", only wells where l.gwe IS TRUE
        elev_min, elev_max        -- filter on l.glo90_elev_m_asl
    """
    clauses = []
    p = {}

    if params.get("lon_min"):
        clauses.append("l.x_wgs84 >= %(lon_min)s")
        p["lon_min"] = float(params["lon_min"])
    if params.get("lon_max"):
        clauses.append("l.x_wgs84 <= %(lon_max)s")
        p["lon_max"] = float(params["lon_max"])
    if params.get("lat_min"):
        clauses.append("l.y_wgs84 >= %(lat_min)s")
        p["lat_min"] = float(params["lat_min"])
    if params.get("lat_max"):
        clauses.append("l.y_wgs84 <= %(lat_max)s")
        p["lat_max"] = float(params["lat_max"])
    if params.get("country"):
        clauses.append("l.country_name = %(country)s")
        p["country"] = params["country"]
    if params.get("elev_min"):
        clauses.append("l.glo90_elev_m_asl >= %(elev_min)s")
        p["elev_min"] = float(params["elev_min"])
    if params.get("elev_max"):
        clauses.append("l.glo90_elev_m_asl <= %(elev_max)s")
        p["elev_max"] = float(params["elev_max"])
    if params.get("litho_only", "").lower() == "true":
        clauses.append("l.litho IS TRUE")
    if params.get("gwh_only", "").lower() == "true":
        clauses.append("l.gwh IS TRUE")
    if params.get("gws_only", "").lower() == "true":
        clauses.append("l.gws IS TRUE")
    if params.get("gwe_only", "").lower() == "true":
        clauses.append("l.gwe IS TRUE")
    if params.get("min_months"):
        clauses.append("COALESCE(obs.num_months, 0) >= %(min_months)s")
        p["min_months"] = int(params["min_months"])

    clause_str = (" AND " + " AND ".join(clauses)) if clauses else ""
    return clause_str, p


def _filtered_wells_cte(year_start, year_end, filter_clause):
    """
    CTE producing one row per well with all metadata + num_months.
    Shared across /api/wells, /api/export/csv, /api/export/shp.

    litho, gwh, gws, gwe are boolean flags in _lookup_tb.
    litho_classes_tb has no FK relationship and is not joined.
    """
    return f"""
        WITH obs AS (
            SELECT
                g.id_gerbil,
                SUM({_month_sum_expr()}) AS num_months
            FROM public._gwh_monthly_tb g
            WHERE g.year BETWEEN %(year_start)s AND %(year_end)s
            GROUP BY g.id_gerbil
        ),
        filtered_wells AS (
            SELECT
                l.id_gerbil,
                l.x_wgs84,
                l.y_wgs84,
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
            WHERE 1=1
            {filter_clause}
        )
    """


def get_connection():
    return psycopg2.connect(**DB)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Quick DB connectivity check."""
    try:
        conn = get_connection()
        conn.close()
        return jsonify({"status": "ok", "db": DB["host"]})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 503


@app.route("/api/wells/countries", methods=["GET"])
def get_countries():
    """
    Returns sorted list of distinct country names present in _lookup_tb.
    Used to populate the country filter dropdown in the portal.
    """
    try:
        sql = """
            SELECT DISTINCT country_name
            FROM public._lookup_tb
            WHERE country_name IS NOT NULL
            ORDER BY country_name
        """
        conn = get_connection()
        df = pd.read_sql(sql, conn)
        conn.close()
        return jsonify({"countries": df["country_name"].tolist()})
    except Exception as e:
        log.exception("Error in /api/wells/countries")
        return jsonify({"error": str(e)}), 500


@app.route("/api/wells", methods=["GET"])
def get_wells():
    """
    Returns well metadata for all wells matching the current filter set.
    All aggregation done in SQL. One summary row per well crosses the network.

    Query params:
        year_start, year_end, min_months,
        lon_min, lon_max, lat_min, lat_max,
        country, elev_min, elev_max,
        litho_only, gwh_only, gws_only, gwe_only
    """
    try:
        year_start = int(request.args.get("year_start", 1979))
        year_end   = int(request.args.get("year_end",   2019))
        filter_clause, filter_params = _build_filter_clauses(request.args)

        sql = f"""
            {_filtered_wells_cte(year_start, year_end, filter_clause)}
            SELECT
                id_gerbil       AS "wellID",
                x_wgs84         AS lon,
                y_wgs84         AS lat,
                country_name,
                state_name,
                litho,
                gwh,
                gws,
                gwe,
                orig_elev_m_asl,
                glo90_elev_m_asl,
                num_months
            FROM filtered_wells
            ORDER BY id_gerbil
        """

        params = {"year_start": year_start, "year_end": year_end, **filter_params}

        conn = get_connection()
        df = pd.read_sql(sql, conn, params=params)
        conn.close()

        # Booleans: cast safely (handles None/NaN before bool conversion)
        for col in ["litho", "gwh", "gws", "gwe"]:
            if col in df.columns:
                df[col] = df[col].map(lambda x: bool(x) if x is not None and str(x) != 'nan' else None)

        # Replace all remaining float NaN with None so jsonify produces valid JSON
        df = df.where(df.notna(), other=None)

        return jsonify({"count": len(df), "wells": df.to_dict(orient="records")})

    except Exception as e:
        log.exception("Error in /api/wells")
        return jsonify({"error": str(e)}), 500


@app.route("/api/timeseries/<well_id>", methods=["GET"])
def get_timeseries(well_id):
    """
    Returns the full monthly GWH time series for a single well.
    Only that well's rows are fetched (at most ~40 rows for 1979-2019).
    cm -> m conversion done in SQL.

    Query params: year_start, year_end
    """
    try:
        year_start = int(request.args.get("year_start", 1979))
        year_end   = int(request.args.get("year_end",   2019))

        sql = f"""
            SELECT
                g.year,
                {_month_select_expr()}
            FROM public._gwh_monthly_tb g
            WHERE g.id_gerbil = %(well_id)s
              AND g.year BETWEEN %(year_start)s AND %(year_end)s
            ORDER BY g.year
        """

        conn = get_connection()

        # Also pull the well's metadata in the same connection
        meta_sql = """
            SELECT
                l.x_wgs84, l.y_wgs84, l.country_name, l.state_name,
                l.glo90_elev_m_asl,
                l.litho, l.gwh, l.gws, l.gwe
            FROM public._lookup_tb l
            WHERE l.id_gerbil = %(well_id)s
        """
        meta_df = pd.read_sql(meta_sql, conn, params={"well_id": well_id})
        df = pd.read_sql(sql, conn, params={
            "well_id": well_id,
            "year_start": year_start,
            "year_end": year_end,
        })
        conn.close()

        if df.empty:
            return jsonify({
                "error": f"No GWH data for well {well_id} in {year_start}-{year_end}"
            }), 404

        # Melt to long format: only ~40 rows x 12 cols, trivial in Python
        month_cols = {f"{m:02d}_gw_head_m": str(m) for m in range(1, 13)}
        df.rename(columns=month_cols, inplace=True)

        melted = pd.melt(
            df,
            id_vars=["year"],
            value_vars=[str(m) for m in range(1, 13)],
            var_name="month",
            value_name="gwh_m",
        )
        melted = melted.dropna(subset=["gwh_m"])
        melted["date"] = (
            melted["year"].astype(str) + "-" + melted["month"].str.zfill(2)
        )
        melted = melted.sort_values("date")
        records = melted[["date", "gwh_m"]].to_dict(orient="records")

        meta = meta_df.to_dict(orient="records")[0] if not meta_df.empty else {}

        return jsonify({
            "wellID": well_id,
            "n": len(records),
            "meta": meta,
            "series": records,
        })

    except Exception as e:
        log.exception("Error in /api/timeseries")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """
    Streams a long-format CSV of all GWH observations for the filtered well set.
    Schema: wellID, lon, lat, country, state, litho_short, elev_m, date, gwh_m

    Uses a named server-side psycopg2 cursor (itersize=2000) so the full
    observation table is never loaded into RAM.

    Query params: same as /api/wells
    """
    try:
        year_start = int(request.args.get("year_start", 1979))
        year_end   = int(request.args.get("year_end",   2019))
        filter_clause, filter_params = _build_filter_clauses(request.args)
        params = {"year_start": year_start, "year_end": year_end, **filter_params}

        obs_sql = f"""
            {_filtered_wells_cte(year_start, year_end, filter_clause)}
            SELECT
                fw.id_gerbil        AS "wellID",
                fw.x_wgs84          AS lon,
                fw.y_wgs84          AS lat,
                fw.country_name,
                fw.state_name,
                fw.litho,
                fw.gwh,
                fw.gws,
                fw.gwe,
                fw.glo90_elev_m_asl AS elev_m,
                g.year,
                {_month_select_expr()}
            FROM public._gwh_monthly_tb g
            JOIN filtered_wells fw ON fw.id_gerbil = g.id_gerbil
            WHERE g.year BETWEEN %(year_start)s AND %(year_end)s
            ORDER BY fw.id_gerbil, g.year
        """

        def generate():
            conn = get_connection()
            try:
                yield "wellID,lon,lat,country,state,litho,gwh,gws,gwe,elev_m,date,gwh_m\n"
                with conn.cursor(
                    name="csv_export_cursor",
                    cursor_factory=psycopg2.extras.DictCursor
                ) as cur:
                    cur.itersize = 2000
                    cur.execute(obs_sql, params)
                    for row in cur:
                        well    = row["wellID"]
                        lon     = row["lon"]
                        lat     = row["lat"]
                        country = row["country_name"] or ""
                        state   = row["state_name"] or ""
                        litho   = str(row["litho"]).lower() if row["litho"] is not None else ""
                        gwh     = str(row["gwh"]).lower() if row["gwh"] is not None else ""
                        gws     = str(row["gws"]).lower() if row["gws"] is not None else ""
                        gwe     = str(row["gwe"]).lower() if row["gwe"] is not None else ""
                        elev    = row["elev_m"] if row["elev_m"] is not None else ""
                        year    = row["year"]
                        for month in range(1, 13):
                            val = row[f"{month:02d}_gw_head_m"]
                            if val is not None:
                                yield (
                                    f"{well},{lon},{lat},{country},{state},"
                                    f"{litho},{gwh},{gws},{gwe},{elev},"
                                    f"{year}-{month:02d},{val:.4f}\n"
                                )
            finally:
                conn.close()

        return Response(
            generate(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=gw_observations.csv"
            },
        )

    except Exception as e:
        log.exception("Error in /api/export/csv")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/shp", methods=["GET"])
def export_shp():
    """
    Streams a zipped shapefile of well locations with all metadata attributes.
    One row per well, no observation rows loaded.
    Schema: wellID, lon, lat, country, state, litho, litho_name, litho_short,
            gwh, gws, gwe, orig_elev, glo90_elev, num_months
    CRS: EPSG:4326

    Query params: same as /api/wells
    """
    try:
        year_start = int(request.args.get("year_start", 1979))
        year_end   = int(request.args.get("year_end",   2019))
        filter_clause, filter_params = _build_filter_clauses(request.args)

        sql = f"""
            {_filtered_wells_cte(year_start, year_end, filter_clause)}
            SELECT
                id_gerbil       AS "wellID",
                x_wgs84         AS lon,
                y_wgs84         AS lat,
                country_name,
                state_name,
                litho,
                gwh,
                gws,
                gwe,
                orig_elev_m_asl,
                glo90_elev_m_asl,
                num_months
            FROM filtered_wells
            ORDER BY id_gerbil
        """

        params = {"year_start": year_start, "year_end": year_end, **filter_params}

        conn = get_connection()
        df = pd.read_sql(sql, conn, params=params)
        conn.close()

        for col in ["gwh", "gws", "gwe"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = os.path.join(tmp, "gw_wells.shp")
            gdf.to_file(shp_path, driver="ESRI Shapefile")

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(tmp):
                    zf.write(os.path.join(tmp, fname), fname)
            zip_buf.seek(0)

        return send_file(
            zip_buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="gw_wells.zip",
        )

    except Exception as e:
        log.exception("Error in /api/export/shp")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)