"""
TEXT HUMANIZER PIPELINE — Three-Expert Architecture (Length-Controlled)
=======================================================================
Drop each section into its own Jupyter cell.
Uses Meta-Llama-3.3-70B-Instruct via HuggingFace Transformers.
Three independent expert rewrites → Discussion → Converged final output.
Word-count anchored to prevent text inflation.
"""

# ============================================================
# CELL 1 — Imports
# ============================================================

from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import BitsAndBytesConfig
import re
import gc

# ============================================================
# CELL 2 — Model Loading
# ============================================================

model_id = '/panfs/g52-panfs/exp/FY25/models/Meta-Llama-3.3-70B-Instruct'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map='auto',
)
print(f"Model loaded on {model.device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# CELL 3 — Text Generation Pipeline
# ============================================================

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    return_full_text=False,
    temperature=0.65,
    max_new_tokens=4096,
    repetition_penalty=1.08,
    do_sample=True,
    top_p=0.9,
)

llm = HuggingFacePipeline(pipeline=generate_text)

# ============================================================
# CELL 4 — Three-Expert Humanization Prompt
# ============================================================

humanize_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are running a humanization process to rewrite AI-generated text so that it appears human-written or human-compiled.

ABSOLUTE LENGTH RULE: The input text has {word_count} words. Each expert rewrite and the final output MUST have between {min_words} and {max_words} words. Any rewrite outside this range is a FAILURE. Do not pad, do not elaborate, do not add examples or new points. If you need to rephrase something longer, shorten something else to compensate.

LANGUAGE RULE: Write in the EXACT same language as the input. If input is French, all rewrites and discussion must be in French. If Arabic, everything in Arabic. Never switch languages.

There are THREE independent experts.
Each expert rewrites the text separately using ONLY the humanization guidelines below.
Experts must not see or reference each other's rewrites during this phase.

Humanization guidelines:
- Vary sentence length and rhythm. Mix short punchy sentences with longer flowing ones. AI text makes every sentence medium-length — break that pattern.
- Break parallel structures. If three sentences start the same way or follow the same pattern, restructure at least one of them differently.
- Remove robotic transitions: eliminate "Furthermore," "Moreover," "Additionally," "It is worth noting that." Either use natural connectors or start sentences directly.
- Use implicit cohesion over explicit signaling. Not every idea needs a transition word linking it to the previous one.
- Add subtle imperfections: a slightly informal phrase, an em-dash, a rhetorical question, a sentence fragment. Real writing is not perfectly consistent in register.
- Vary paragraph length. Not every paragraph should be the same size.
- Do NOT add new information or points not in the original.
- Do NOT remove information or meaning from the original.
- Do NOT summarize or condense the original.
- Do NOT improve clarity or elegance — aim for natural, not polished.
- STAY WITHIN {min_words}–{max_words} WORDS. This is non-negotiable.

Expert roles:
Expert 1: Focuses on sentence-level irregularity and pacing. Varies sentence lengths aggressively.
Expert 2: Focuses on structure, formatting, and list behavior. Breaks up any AI-typical patterns like uniform paragraphs or bullet-point-like structures.
Expert 3: Focuses on reducing AI-like regularity at the global level. Changes how ideas connect, varies register slightly, makes the overall flow feel less systematic.

Each expert must output:
- A full rewritten version of the text (same language as input).
- No explanations, commentary, labels, or annotations.
- Word count must be between {min_words} and {max_words}.

Discussion phase:
After all rewrites are produced, conduct a BRIEF discussion (3–4 exchanges max) where:
- Experts compare rewrites and identify where AI-like regularity still remains.
- Experts may borrow localized changes from other rewrites.
- The goal is convergence, not merging everything together. Pick the BEST phrasing for each section, not the longest.
- If any version is too long, the discussion must address trimming.

Final output:
Produce ONE consolidated rewritten version that:
- Preserves the original style, content, and meaning.
- Incorporates human-like imperfections.
- Does not appear optimized or machine-polished.
- Is between {min_words} and {max_words} words. CHECK THIS BEFORE OUTPUTTING.
- Show the discussion of the judges and how they came up with the final output.

Restrictions:
- Do NOT explain the changes.
- Do NOT describe reasoning or internal deliberation outside the Discussion phase.
- The final output section must contain ONLY the final rewritten text after the discussion.

Rewrite the following text:
{input_text} <|eot_id|><|start_header_id|>user<|end_header_id|>'''

# ============================================================
# CELL 5 — Quality Check / Trim Prompt (Pass 2 — Optional)
# ============================================================

analysis_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a panel of three expert human editors. Each editor is fluent in the language of the manuscript being edited (e.g., English, French, German, Spanish, Arabic, Farsi, etc.).

Input: You will be given a manuscript that was humanized from AI-generated text. It may still have some AI-like patterns.

CRITICAL LENGTH CONSTRAINT: The manuscript should be approximately {word_count} words. If it is longer than {max_words} words, you MUST trim it back to approximately {word_count} words by tightening phrasing. NEVER add content. NEVER expand.

The editors discuss the manuscript internally in the INPUT LANGUAGE, identifying patterns typical of AI-generated text.
They collaboratively decide how to revise the text to better reflect authentic human writing practices.

Objective: Rewrite the manuscript in its original language so it is indistinguishable from human-written text.

Humanization & Anti-Detection Constraints:
- Replace formulaic transitions ("Furthermore," "Moreover," "In addition") with natural alternatives or remove them entirely.
- Vary sentence openings — avoid starting consecutive sentences the same way.
- If the text feels too polished or uniform, roughen it up slightly.
- Preserve ALL original meaning and content.
- Do NOT add new information.
- Output MUST be between {min_words} and {max_words} words.

Prioritize human plausibility over stylistic polish.
If a sentence sounds "too perfect," rewrite it.
Prefer implicit cohesion over explicit signaling.

Output ONLY the final rewritten text. No commentary, no labels, no explanations, no word counts.

Rewrite the following text: {input_text}

<|eot_id|><|start_header_id|>user<|end_header_id|>'''

# ============================================================
# CELL 6 — Helper Functions
# ============================================================

def count_words(text):
    """Count words in text. Works for most Latin/Arabic/Cyrillic scripts."""
    return len(text.split())


def clean_output(text):
    """
    Extract the final rewritten version from model output.
    Handles the three-expert format by grabbing content after
    the last 'Version finale' / 'Final output' / 'Final version' marker.
    """
    # Try to find the final consolidated output
    # Look for common markers the model uses after discussion
    final_markers = [
        r'\*\*Version [Ff]inale\*\*',
        r'\*\*Final [Oo]utput\*\*',
        r'\*\*Final [Vv]ersion\*\*',
        r'\*\*Consolidated [Vv]ersion\*\*',
        r'\*\*Texte [Ff]inal\*\*',
        r'\*\*Résultat [Ff]inal\*\*',
        r'Version [Ff]inale\s*:',
        r'Final [Oo]utput\s*:',
        r'Final [Vv]ersion\s*:',
    ]
    
    for marker in final_markers:
        match = re.search(marker, text)
        if match:
            after = text[match.end():].strip()
            # Clean up any remaining markers or whitespace
            after = re.sub(r'^\s*:\s*', '', after).strip()
            after = re.sub(r'^\s*\n', '', after).strip()
            if len(after) > 50:  # sanity check
                return after.strip()
    
    # If no marker found, return full text (fallback)
    # But strip common preamble patterns
    lines = text.strip().split('\n')
    skip_patterns = [
        r'^\*\*Expert\s*\d',
        r'^Expert\s*\d',
        r'^here\s*(is|\'s)',
        r'^voici',
        r'^the rewritten',
    ]
    
    # Just return everything — the model didn't use expected markers
    cleaned = text.strip()
    
    # Remove trailing meta-commentary
    trailing = [
        r'\n\s*\*\*Note:.*$',
        r'\n\s*Note:.*$',
        r'\n\s*Word count:.*$',
        r'\n\s*\(Word count:.*$',
        r'\n\s*\(\d+ words?\).*$',
    ]
    for pattern in trailing:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    return cleaned.strip()


def extract_final_only(text):
    """
    More aggressive extraction: tries to get ONLY the final version
    after the discussion phase, stripping expert rewrites and discussion.
    Use this if clean_output still returns too much.
    """
    # Split on "Version finale" or equivalent and take the last chunk
    splits = re.split(
        r'\*\*(?:Version [Ff]inale|Final [Oo]utput|Final [Vv]ersion|Texte [Ff]inal)\*\*',
        text
    )
    if len(splits) > 1:
        final = splits[-1].strip()
        final = re.sub(r'^\s*:\s*', '', final).strip()
        # Remove any trailing notes
        final = re.sub(r'\n\s*\*\*.*$', '', final, flags=re.DOTALL).strip()
        if len(final) > 50:
            return final
    
    return clean_output(text)


# ============================================================
# CELL 7 — Main Humanization Function
# ============================================================

def humanize(input_text, do_polish_pass=True, tolerance=0.10, verbose=True):
    """
    Run the three-expert humanization pipeline.
    
    Args:
        input_text: AI-generated text in any language
        do_polish_pass: Run the second analysis/trim pass (recommended)
        tolerance: Allowed word count deviation (default ±10%)
        verbose: Print progress info
    
    Returns:
        dict with:
            'result'       — final humanized text
            'full_output'  — raw model output (includes experts + discussion)
            'input_words'  — input word count
            'output_words' — output word count
            'ratio'        — output/input word ratio
    """
    wc = count_words(input_text)
    min_w = int(wc * (1 - tolerance))
    max_w = int(wc * (1 + tolerance))
    
    if verbose:
        print(f"Input word count: {wc}")
        print(f"Target range: {min_w}–{max_w} words")
        print(f"Running three-expert humanization...")
    
    # --- Pass 1: Three-expert rewrite ---
    prompt_1 = PromptTemplate(
        input_variables=["input_text", "word_count", "min_words", "max_words"],
        template=humanize_template,
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)
    
    result_1 = chain_1.invoke({
        "input_text": input_text,
        "word_count": str(wc),
        "min_words": str(min_w),
        "max_words": str(max_w),
    })
    
    raw_output = result_1['text']
    final_text = extract_final_only(raw_output)
    out_wc = count_words(final_text)
    
    if verbose:
        print(f"Pass 1 — extracted final: {out_wc} words (ratio: {out_wc/wc:.2f}x)")
    
    # --- Pass 2: Quality check + trim (if enabled) ---
    if do_polish_pass:
        if verbose:
            reason = "length exceeded" if out_wc > max_w else "quality polish"
            print(f"Running pass 2 ({reason})...")
        
        prompt_2 = PromptTemplate(
            input_variables=["input_text", "word_count", "min_words", "max_words"],
            template=analysis_template,
        )
        chain_2 = LLMChain(llm=llm, prompt=prompt_2)
        
        result_2 = chain_2.invoke({
            "input_text": final_text,
            "word_count": str(wc),
            "min_words": str(min_w),
            "max_words": str(max_w),
        })
        
        final_text = clean_output(result_2['text'])
        out_wc = count_words(final_text)
        
        if verbose:
            print(f"Pass 2 output: {out_wc} words (ratio: {out_wc/wc:.2f}x)")
    
    ratio = out_wc / wc
    in_range = min_w <= out_wc <= max_w
    status = "PASS" if in_range else f"WARN: {out_wc} words (target {min_w}–{max_w})"
    
    if verbose:
        print(f"Final: {out_wc} words | {status}")
    
    return {
        "result": final_text,
        "full_output": raw_output,
        "input_words": wc,
        "output_words": out_wc,
        "ratio": ratio,
    }


# ============================================================
# CELL 8 — Run It
# ============================================================

# Paste your input text here (any language)
english_text = '''Chris Noth et Sarah Jessica Parker restent indissociables de l'histoire de la télévision contemporaine grâce à leurs rôles respectifs de Mr. Big et Carrie Sarah Jessica Parker, également productrice exécutive de la série et de ses prolongements cinématographiques, est devenue une icône mondiale, tant pour son talent d'actrice que pour son influence sur la mode. Elle a contribué à redéfinir les rôles féminins à la télévision en jouant Carrie Bradshaw, un personnage vulnérable, indépendant et ambitieux. Chris Noth, de son côté, a incarné un personnage complexe et parfois controversé, Mr. Big, symbole des relations amoureuses imparfaites mais passionnelles. La relation à l'écran entre Noth et Parker a connu de nombreux rebondissements, reflétant les hésitations et contradictions de l'amour moderne. Leur alchimie à l'écran entre Noth et Parker a connu de nombreux rebondissements, reflétant les hésitations et contradictions de l'amour moderne. Aujourd'hui encore, l'héritage de Sex and the City continue d'alimenter débats, analyses et nostalgie, preuve que le duo formé par Chris Noth et Sarah Jessica Parker a marqué durablement la culture populaire.'''

output = humanize(english_text, do_polish_pass=True, tolerance=0.10)

print("\n" + "="*60)
print("HUMANIZED OUTPUT:")
print("="*60)
print(output["result"])

# To see the full expert rewrites + discussion:
# print(output["full_output"])


# ============================================================
# CELL 9 — Batch Processing (Optional)
# ============================================================

def humanize_batch(texts, do_polish_pass=True, tolerance=0.10):
    """Process multiple texts. Returns list of result dicts."""
    results = []
    for i, text in enumerate(texts):
        print(f"\n{'='*40}")
        print(f"Processing text {i+1}/{len(texts)}")
        print(f"{'='*40}")
        r = humanize(text, do_polish_pass=do_polish_pass, tolerance=tolerance)
        results.append(r)
        gc.collect()
    return results

# Example:
# texts = [text1, text2, text3]
# results = humanize_batch(texts)
# for r in results:
#     print(r["result"])
#     print(f"Ratio: {r['ratio']:.2f}x\n")
