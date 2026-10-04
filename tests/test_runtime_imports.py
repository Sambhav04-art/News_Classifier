"""Guard: the serving code must not depend on training-only packages.

Render installs only requirements.txt (no pandas/matplotlib), so importing
them from the API path would crash the deployment at startup.
"""
import subprocess
import sys
import textwrap


def test_api_works_without_training_packages():
    code = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class Block(MetaPathFinder):
            # Makes the packages truly absent, like on Render
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("pandas", "matplotlib"):
                    raise ModuleNotFoundError(f"No module named {name!r}")

        sys.meta_path.insert(0, Block())
        import app.main  # noqa: F401
        from src.predict import NewsClassifier
        assert NewsClassifier().predict("Stocks rally on Wall Street")["category"]
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
