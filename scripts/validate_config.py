"""
Validate a configuration file and report any issues.

Usage:
    python scripts/validate_config.py configs/baseline_config.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulationParameters


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_config.py <config_file.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"Validating configuration: {config_path}")
    print("=" * 70)

    try:
        params = SimulationParameters.from_yaml(config_path)
        print("[OK] Configuration loaded successfully")
        print()

        # Run validation
        warnings = params.validate()

        if not warnings:
            print("[OK] All validation checks passed!")
            print()
            print("Configuration summary:")
            print(params)
            sys.exit(0)
        else:
            # Categorize warnings
            errors = [w for w in warnings if w.startswith("ERROR")]
            warns = [w for w in warnings if w.startswith("WARNING")]
            infos = [w for w in warnings if w.startswith("INFO")]

            if errors:
                print(f"[ERROR] {len(errors)} ERROR(S) found:")
                for error in errors:
                    print(f"  {error}")
                print()

            if warns:
                print(f"[WARN] {len(warns)} WARNING(S):")
                for warn in warns:
                    print(f"  {warn}")
                print()

            if infos:
                print(f"[INFO] {len(infos)} INFO message(s):")
                for info in infos:
                    print(f"  {info}")
                print()

            if errors:
                print("Configuration has ERRORS and should not be used for simulation.")
                sys.exit(1)
            else:
                print("Configuration has warnings but may be usable.")
                print("Review warnings carefully before running simulation.")
                sys.exit(0)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
