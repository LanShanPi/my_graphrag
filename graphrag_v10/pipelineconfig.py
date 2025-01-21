# config.py

class PipelineConfig:
    """
    用于控制整个管线：是否执行符号推理、使用何种推理方式、chunk大小、是否调试等。
    也可以扩展更多选项，比如“是否执行多文档汇总”、“是否使用特定LLM”等。
    """
    def __init__(
        self,
        enable_symbolic_reasoning=False,
        reasoning_mode="owl",  # "owl" / "prolog" / "sympy" / ...
        chunk_size=128,
        overlap=10,
        max_triplets_per_chunk=10,
        debug=False
    ):
        self.enable_symbolic_reasoning = enable_symbolic_reasoning
        self.reasoning_mode = reasoning_mode
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.debug = debug