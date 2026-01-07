import argparse
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fund_config", required=True, help="Path to fund config JSON")
    parser.add_argument("--holdings_csv", required=True, help="Path to holdings CSV")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", default="runs", help="Output directory")
    args = parser.parse_args()

    out = run_pipeline(
        fund_config_path=args.fund_config,
        holdings_csv_path=args.holdings_csv,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.out
    )

    print("✅ Run complete:", out["run_dir"])
    print(out["summary"])


if __name__ == "__main__":
    main()
