# BioASQ Appendix — Putting Prompts in Boxes

Two options. **Use Option A** unless you specifically want rounded corners. Option A uses `listings`, which you already load, so it's the safe choice on the CEUR template.

---

## OPTION A — Framed grey boxes (recommended)

### Step 1 — Add to your PREAMBLE

Paste this right after your existing `\lstset{breaklines=true}` line (around line 20). If you already load `xcolor` elsewhere, delete the first line here to avoid a duplicate-package warning.

```latex
\usepackage{xcolor}
\definecolor{promptbg}{RGB}{246,247,249}
\definecolor{promptframe}{RGB}{200,204,210}
\lstdefinestyle{prompt}{
  backgroundcolor=\color{promptbg},
  frame=single,
  rulecolor=\color{promptframe},
  framesep=6pt,
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  breakindent=0pt,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  xleftmargin=4pt,
  xrightmargin=4pt,
  aboveskip=8pt,
  belowskip=8pt
}
```

### Step 2 — Update each prompt in the appendix

Change every prompt's opening tag from:

```latex
\begin{lstlisting}
```

to:

```latex
\begin{lstlisting}[style=prompt]
```

The closing `\end{lstlisting}` does not change. Do this for all eight prompt blocks (query generation, sufficiency check, yes/no, factoid, list, summary, LLM rerank).

### Example of one finished prompt block

```latex
\paragraph{Factoid generation.}
\begin{lstlisting}[style=prompt]
You are an expert biomedical QA system.

STRICT RULES:
1. EXACT_ANSWER must be 1-5 words. A specific name, number, or term.
2. Copy the EXACT terminology from the evidence passages.
3. Do NOT paraphrase, explain, or elaborate in the exact answer.
4. If the evidence does not contain a clear specific answer, write: unknown
5. Prefer named entities: drug names, protein names, gene symbols,
   disease names, specific numbers/percentages.
6. Then write IDEAL_ANSWER: a 2-4 sentence explanation (max 200 words).

GOOD: 'transsphenoidal surgery', 'NF1', '45,X', 'palivizumab', 'mesenchymal'
BAD:  'multiple causative factors', 'a type of bleeding'

[two few-shot examples]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
EXACT_ANSWER:
\end{lstlisting}
```

---

## OPTION B — Rounded boxes (prettier, adds one package)

Use this only if you want rounded corners. It adds `tcolorbox`, which occasionally conflicts with CEUR-ART float handling — slightly more risk this close to camera-ready.

### Step 1 — Add to your PREAMBLE

```latex
\usepackage[most]{tcolorbox}
\newtcblisting{promptbox}{
  listing only,
  colback=gray!4,
  colframe=gray!40,
  boxrule=0.4pt,
  arc=2pt,
  left=4pt, right=4pt, top=3pt, bottom=3pt,
  listing options={
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    columns=fullflexible,
    keepspaces=true
  }
}
```

### Step 2 — Replace each prompt's wrapper

Change `\begin{lstlisting}` → `\begin{promptbox}` and `\end{lstlisting}` → `\end{promptbox}` for every prompt.

### Example

```latex
\paragraph{Factoid generation.}
\begin{promptbox}
You are an expert biomedical QA system.

STRICT RULES:
1. EXACT_ANSWER must be 1-5 words. A specific name, number, or term.
...
EXACT_ANSWER:
\end{promptbox}
```

---

## If a box breaks ugly across a page

After compiling, if any prompt box splits awkwardly over a page boundary, wrap that prompt's `\paragraph{...}` plus its box in a floating figure so LaTeX pushes it to a clean page top:

```latex
\begin{figure}[t]
\paragraph{List generation.}
\begin{lstlisting}[style=prompt]
... prompt text ...
\end{lstlisting}
\end{figure}
```

Only do this for the specific prompt that breaks badly — don't wrap all of them by default.

---

## Recompile

After the preamble change and the tag updates, recompile twice. All eight prompts should now sit in matching boxes, clearly separated from the body text.
