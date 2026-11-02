"""Core DCT compression algorithms and mathematical explanations."""

from .blocks import merge_blocks, split_blocks
from .dct import BLOCK, D, dct2, dct2_batch, idct2, idct2_batch
from .explanation import (
    STANDARD_SAMPLE_BLOCK,
    BlockCalculationSteps,
    calculate_block_steps,
    format_block_walkthrough,
    format_matrix,
    generate_full_academic_report,
    get_algorithms_text,
    get_formulas_text,
    get_full_image_metrics_breakdown,
    get_pipeline_flowchart,
)
from .quant import (
    LUM_TABLE_Q50,
    dequantize,
    quantize,
    scale_table,
)

__all__ = [
    "BLOCK",
    "D",
    "dct2",
    "dct2_batch",
    "idct2",
    "idct2_batch",
    "split_blocks",
    "merge_blocks",
    "LUM_TABLE_Q50",
    "scale_table",
    "quantize",
    "dequantize",
    "STANDARD_SAMPLE_BLOCK",
    "BlockCalculationSteps",
    "calculate_block_steps",
    "format_matrix",
    "get_pipeline_flowchart",
    "get_algorithms_text",
    "get_formulas_text",
    "format_block_walkthrough",
    "get_full_image_metrics_breakdown",
    "generate_full_academic_report",
]
