# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pathlib
from dataclasses import dataclass

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()

EXAMPES_DIR = PROJECT_DIR / "examples"

# The path to the exporter module directory.
EXPORTER_ROOT_DIR = PROJECT_DIR / "exporter"

EXPORTER_TESTS_DIR = EXPORTER_ROOT_DIR / "tests"

# ONNX file extension
ONNX_EXTENSION = ".onnx"


def _ensure_onnx_extension(filename: str) -> pathlib.Path:
    """Ensure filename has .onnx extension.

    Args:
        filename: Name of the ONNX file.

    Returns:
        Path object with correct extension.
    """
    file_name = pathlib.Path(filename)
    if file_name.suffix != ONNX_EXTENSION:
        file_name = file_name.with_suffix(ONNX_EXTENSION)
    return file_name


def _ensure_debug_dir(base_dir: pathlib.Path) -> pathlib.Path:
    """Create and return debug directory path.

    Args:
        base_dir: Base directory for the debug folder.

    Returns:
        Path to the debug directory.
    """
    debug_path = base_dir / "debug"
    debug_path.mkdir(parents=True, exist_ok=True)
    return debug_path


@dataclass
class OnnxPaths:
    """Container for ONNX file paths."""

    main: pathlib.Path
    """Path to the main ONNX file."""

    debug_dir: pathlib.Path
    """Path to the debug directory."""

    debug_variants: dict[str, pathlib.Path]
    """Dictionary of debug variant names to their paths."""

    def get_debug_path(self, suffix: str) -> pathlib.Path:
        """Get a debug variant path by suffix."""
        return self.debug_variants.get(suffix)


def prepare_onnx_paths(
    output_dir: str | pathlib.Path,
    filename: str,
    debug_suffixes: list[str] | None = None,
) -> OnnxPaths:
    """Prepare and validate paths for ONNX files.

    This function:
    - Ensures the filename has .onnx extension
    - Creates the debug directory
    - Returns structured paths for main and debug model files

    Args:
        output_dir: Directory where the ONNX file will be saved/loaded.
        filename: Name of the ONNX file (extension will be added if missing).
        debug_suffixes: Optional list of debug variant suffixes (e.g., ['default', 'optimized']).

    Returns:
        OnnxPaths object containing all required paths.
    """
    path = pathlib.Path(output_dir)
    file_name = _ensure_onnx_extension(filename)
    debug_path = _ensure_debug_dir(path)

    # Generate debug variant paths
    debug_variants = {}
    if debug_suffixes:
        for suffix in debug_suffixes:
            debug_variants[suffix] = debug_path / f"{file_name.stem}_{suffix}{ONNX_EXTENSION}"

    return OnnxPaths(
        main=path / file_name,
        debug_dir=debug_path,
        debug_variants=debug_variants,
    )
