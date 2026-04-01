"""
Diagnostic script to verify Python environment: numpy, pandas, matplotlib, ta, yfinance.
Prints version for each if import succeeds; otherwise prints the exception.
"""

def _check(name: str, import_func):
    try:
        mod = import_func()
        version = getattr(mod, "__version__", "unknown")
        print(f"{name} version: {version}")
        return True
    except Exception as e:
        print(f"{name}: import failed - {e}")
        return False

def main():
    print("Environment check:\n")
    _check("numpy", lambda: __import__("numpy"))
    _check("pandas", lambda: __import__("pandas"))
    _check("matplotlib", lambda: __import__("matplotlib"))
    _check("ta", lambda: __import__("ta"))
    _check("yfinance", lambda: __import__("yfinance"))
    print("\nDone.")

if __name__ == "__main__":
    main()
