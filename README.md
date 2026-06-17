\begin{table}[t]
\centering
\caption{BioASQ-style validation results}\label{tab:results}
\begin{tabular}{lccccc}
\toprule
System & Fact. & Y/N & List & SemF1 & Ovr.\\
\midrule
Zero-shot Gemma 4 31B & 8.0 & 94.0 & 0.0 & 41.6 & 35.9\\
Agentic SQLite FTS5 & 62.0 & 88.0 & 87.0 & 49.0 & 71.5\\
Agentic hybrid FAISS+BM25+IVF & 69.0 & 88.0 & 96.0 & 51.0 & \textbf{76.0}\\
Agentic SQLite FTS5, CoT & 50.0 & 82.0 & 70.0 & 49.0 & 62.75\\
DSPy RAG & 12.0 & 82.0 & 4.0 & 49.0 & 36.75\\
DSPy RAG + MIPROv2 & 27.0 & 94.0 & 13.0 & 48.0 & 45.5\\
DSPy MultiHop RAG & 12.0 & 76.0 & 0.0 & 49.0 & 34.25\\
DSPy MultiHop RAG + MIPROv2 & 27.0 & 88.0 & 13.0 & 48.0 & 44.0\\
DSPy ReAct RAG & 8.0 & 76.0 & 0.0 & 55.0 & 34.75\\
DSPy ReAct RAG + MIPROv2 & 31.0 & 94.0 & 0.0 & 55.0 & 45.0\\
\bottomrule
\end{tabular}
\end{table}


\usepackage{booktabs}
