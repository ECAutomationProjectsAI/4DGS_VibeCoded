#!/usr/bin/env python3
"""4D Gaussian Splatting format converter CLI tool.

Convert between different 4DGS representations:
- gs4d (our format)
- SpacetimeGaussian (PLY with timestamps)
- Fudan 4DGS (with deformation fields)

Usage:
    python convert.py input.ply output.pt --from spacetime --to gs4d
    python convert.py checkpoint.pth output.ply --from fudan --to spacetime
"""

import os
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gs4d.converters import (
    BaseConverter,
    SpacetimeGaussianConverter,
    FudanConverter
)


def get_converter(format_name: str) -> BaseConverter:
    """Get converter instance for specified format."""
    converters = {
        'spacetime': SpacetimeGaussianConverter,
        'spacetimegaussian': SpacetimeGaussianConverter,
        'stg': SpacetimeGaussianConverter,
        'fudan': FudanConverter,
        'fudan4dgs': FudanConverter,
        'f4dgs': FudanConverter,
    }
    
    format_lower = format_name.lower()
    if format_lower not in converters:
        raise ValueError(f"Unknown format: {format_name}. Supported: {list(converters.keys())}")
    
    return converters[format_lower]()


def convert(
    input_path: str,
    output_path: str,
    from_format: str,
    to_format: str,
    chunk_size: int = 10000,
    verbose: bool = True
):
    """
    Convert between 4DGS formats.
    
    Args:
        input_path: Path to input file/directory
        output_path: Path to output file/directory
        from_format: Source format name
        to_format: Target format name
        chunk_size: Number of Gaussians to process at once
        verbose: Print progress messages
    """
    
    # Validate paths
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Handle conversion
    if from_format.lower() == 'gs4d' and to_format.lower() == 'gs4d':
        # Just copy
        import shutil
        shutil.copy2(input_path, output_path)
        if verbose:
            print(f"Copied gs4d file: {input_path} -> {output_path}")
        return
    
    if from_format.lower() == 'gs4d':
        # Convert from gs4d to another format
        converter = get_converter(to_format)
        converter.chunk_size = chunk_size
        
        if verbose:
            print(f"Converting from gs4d to {to_format}...")
        
        converter.convert_from_gs4d(input_path, output_path)
        
    elif to_format.lower() == 'gs4d':
        # Convert from another format to gs4d
        converter = get_converter(from_format)
        converter.chunk_size = chunk_size
        
        if verbose:
            print(f"Converting from {from_format} to gs4d...")
        
        converter.convert_to_gs4d(input_path, output_path)
        
    else:
        # Convert between two non-gs4d formats (go through gs4d as intermediate)
        import tempfile
        
        if verbose:
            print(f"Converting from {from_format} to {to_format} (via gs4d)...")
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # First convert to gs4d
            source_converter = get_converter(from_format)
            source_converter.chunk_size = chunk_size
            source_converter.convert_to_gs4d(input_path, tmp_path)
            
            # Then convert from gs4d to target
            target_converter = get_converter(to_format)
            target_converter.chunk_size = chunk_size
            target_converter.convert_from_gs4d(tmp_path, output_path)
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    if verbose:
        print(f"Conversion complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert between 4D Gaussian Splatting formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  gs4d       - Our unified 4DGS format (.pt/.pth files)
  spacetime  - SpacetimeGaussian PLY format with timestamps
  fudan      - Fudan 4DGS with deformation fields

Examples:
  # Convert SpacetimeGaussian PLY to our format
  python convert.py input.ply output.pt --from spacetime --to gs4d

  # Convert our format to Fudan 4DGS
  python convert.py model.pt fudan_model/ --from gs4d --to fudan

  # Convert between external formats
  python convert.py spacetime.ply fudan.pth --from spacetime --to fudan

  # Batch conversion (requires custom script)
  for file in *.ply; do
    python convert.py "$file" "${file%.ply}.pt" --from spacetime --to gs4d
  done
        """
    )
    
    parser.add_argument('input', help='Input file or directory path')
    parser.add_argument('output', help='Output file or directory path')
    parser.add_argument('--from', dest='from_format', required=True,
                       choices=['gs4d', 'spacetime', 'stg', 'fudan', 'f4dgs'],
                       help='Source format')
    parser.add_argument('--to', dest='to_format', required=True,
                       choices=['gs4d', 'spacetime', 'stg', 'fudan', 'f4dgs'],
                       help='Target format')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Number of Gaussians to process at once (for memory efficiency)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    try:
        convert(
            input_path=args.input,
            output_path=args.output,
            from_format=args.from_format,
            to_format=args.to_format,
            chunk_size=args.chunk_size,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()