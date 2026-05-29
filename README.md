def __init__(self, llm, call_graph, src_dir, output_dir,
                 fuzz_seconds=120, main_binary=None, tool_context=""):
        self.scanner = ScannerAgent(llm, max_workers=3)


def __init__(self, llm, call_graph, src_dir, output_dir,
                 fuzz_seconds=120, main_binary=None, tool_context=""):
        self.llm = llm
        self.scanner = ScannerAgent(llm, max_workers=3)
