"""Satu CLI untuk mengunduh, memfilter, dan menganalisis deteksi VIIRS."""

import argparse
import logging
from pathlib import Path

from fire_spot.config import BOUNDARIES_PATH, KALIMANTAN, OUTPUT_ROOT, RAW_DATA, SUMATRA


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fire_spot", description="Unduh dan analisis titik api VIIRS Indonesia."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Unduh VIIRS NOAA-21 tujuh hari terbaru.")
    download.add_argument("--output", type=Path, default=RAW_DATA)

    analyze = commands.add_parser("analyze", help="Analisis cluster; default wilayah Sumatra.")
    analyze.add_argument(
        "--region", default="sumatra", help="sumatra, indonesia, all, atau nama provinsi."
    )
    analyze.add_argument("--clusters", type=int, default=5, help="K untuk metode kmeans.")
    analyze.add_argument("--max-k", type=int, default=10, help="Batas kurva elbow untuk kmeans.")

    filter_data = commands.add_parser("filter", help="Simpan CSV per provinsi.")
    regions = filter_data.add_mutually_exclusive_group()
    regions.add_argument("--regions", nargs="+", help="Nama provinsi, misalnya Jambi Riau.")
    regions.add_argument("--all", action="store_true", help="Semua provinsi dalam batas wilayah.")
    regions.add_argument("--sumatra", action="store_true", help="Sumatra (default).")
    regions.add_argument("--kalimantan", action="store_true")
    filter_data.add_argument("--method", choices=["geojson", "bounding_box"], default="geojson")
    filter_data.add_argument("--no-maps", action="store_true")

    for command in (analyze, filter_data):
        command.add_argument("--data", type=Path, default=RAW_DATA)
        command.add_argument("--boundaries", type=Path, default=BOUNDARIES_PATH)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--basemap", choices=["street", "topo"], default="street")
        command.add_argument("--include-low-confidence", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "download":
            from fire_spot.download import download_latest

            download_latest(args.output)
        elif args.command == "analyze":
            from fire_spot.analysis import run_analysis

            run_analysis(
                args.data,
                region=args.region,
                n_clusters=args.clusters,
                output_root=args.output_dir or OUTPUT_ROOT,
                max_k=args.max_k,
                basemap=args.basemap,
                boundaries_path=args.boundaries,
                include_low_confidence=args.include_low_confidence,
            )
        else:
            from fire_spot.filtering import generate_filtered_datasets

            selected = SUMATRA
            if args.all:
                selected = None
            elif args.kalimantan:
                selected = KALIMANTAN
            elif args.regions:
                selected = args.regions
            generate_filtered_datasets(
                regions_to_filter=selected,
                method=args.method,
                raw_data_path=args.data,
                boundaries_path=args.boundaries,
                output_dir=args.output_dir,
                make_maps=not args.no_maps,
                basemap=args.basemap,
                include_low_confidence=args.include_low_confidence,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0
