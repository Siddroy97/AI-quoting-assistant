import re
import sys
import os
import json
import pathlib
import datetime
import anthropic
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, FileResponse

# Load .env from the project root regardless of where the server is launched from
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so pipeline module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retrieve import retrieve as run_retrieval
from pipeline.retrieve import retrieve_rules as run_retrieve_rules

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLAUDE_MODEL = "claude-sonnet-4-6"
EXTRACTED_FOLDER = pathlib.Path("extracted")
QUOTES_FOLDER = pathlib.Path("quotes")
CORRECTIONS_FILE = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "corrections.json"
ASSIGNMENTS_FILE = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "assignments.json"
HISTORICAL_REFERENCE_FILE = "TMI-Q-2022-011.txt"
FORMULA_TOLERANCE = 0.02
PRICE_TOLERANCE = 0.30
REQUIRED_SECTIONS = [
    "EQUIPMENT SPECIFICATION",
    "BILL OF MATERIALS",
    "PRICING SUMMARY",
    "LEAD TIME",
    "COMMERCIAL TERMS",
    "ENGINEERING NOTES",
]


# Pydantic model for the 6 RFQ input fields
class RFQInput(BaseModel):
    facility_size: str
    cooling_load: str
    config_type: str
    location: str
    redundancy_tier: str
    timeline: str


# Pydantic model for the generate-quote endpoint — RFQ fields plus retrieved documents
class GenerateQuoteInput(RFQInput):
    retrieved_documents: list


# Extracts a named section from document text using numbered section headings as boundaries
def extract_section(document_text, section_keyword):
    pattern = rf"(\d+\.\s+{section_keyword}[^\n]*\n.*?)(?=\n\d+\.\s+[A-Z]|OUTCOME:|$)"
    match = re.search(pattern, document_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# Pulls engineering notes and pricing summary from a single retrieved document's text
def extract_relevant_sections(document_text):
    pricing_summary = extract_section(document_text, "PRICING SUMMARY")
    engineering_notes = extract_section(document_text, "ENGINEERING NOTES")
    sections = []
    if pricing_summary:
        sections.append(f"PRICING SUMMARY:\n{pricing_summary}")
    if engineering_notes:
        sections.append(f"ENGINEERING NOTES:\n{engineering_notes}")
    if not sections:
        return "No relevant sections found."
    return "\n\n".join(sections)


# Builds the prompt for Claude from the RFQ fields, retrieved documents, and optional engineering rules
def build_generation_prompt(rfq, retrieved_documents, engineering_rules=None):
    rfq_summary = (
        f"Facility size: {rfq.facility_size}\n"
        f"Cooling load: {rfq.cooling_load}\n"
        f"Configuration type: {rfq.config_type}\n"
        f"Location: {rfq.location}\n"
        f"Redundancy tier: {rfq.redundancy_tier}\n"
        f"Timeline: {rfq.timeline}"
    )

    context_blocks = []
    for index, document in enumerate(retrieved_documents, start=1):
        document_text = document.get("text", "")
        filename = document.get("filename", f"Document {index}")
        similarity = document.get("similarity", "N/A")
        relevant_sections = extract_relevant_sections(document_text)
        context_blocks.append(
            f"--- Historical Quote {index}: {filename} (Similarity: {similarity}%) ---\n"
            f"{relevant_sections}"
        )

    context_text = "\n\n".join(context_blocks)

    rules_block = ""
    if engineering_rules:
        rules_lines = "\n".join(f"- {rule}" for rule in engineering_rules)
        rules_block = f"\nENGINEERING RULES FROM KNOWLEDGE BASE:\n{rules_lines}\n\n"

    return (
        "You are a quoting assistant for TMI Climate Solutions, a company that builds custom "
        "air handling and cooling systems for data centres.\n\n"
        "Below is a new Request for Quotation (RFQ), followed by three similar historical quotes "
        "from TMI's archive. Use the historical quotes as reference for pricing structure, "
        "engineering approach, and language style.\n\n"
        "Synthesise a draft quote that:\n"
        "- Follows TMI's pricing formula: Overhead = (Materials + Labour) x 18%, "
        "Total = Pre-margin subtotal / 0.78\n"
        "- Uses exactly these numbered sections in this order, with these exact heading names:\n"
        "  1. PROJECT SCOPE SUMMARY\n"
        "  2. EQUIPMENT SPECIFICATION\n"
        "  3. BILL OF MATERIALS\n"
        "  4. PRICING SUMMARY\n"
        "  5. LEAD TIME AND SCHEDULE\n"
        "  6. COMMERCIAL TERMS\n"
        "  7. ENGINEERING NOTES\n"
        "- Formats each section heading as ## N. SECTION NAME exactly, "
        "where N is the section number\n"
        "- Structures each note in Section 7 as a one-sentence lead followed by "
        "a maximum of 4 bullet sub-points for supporting detail\n"
        "- Makes reasonable engineering assumptions based on the RFQ inputs and historical context\n"
        "- Flags any assumptions clearly in the engineering notes section\n\n"
        f"NEW RFQ:\n{rfq_summary}\n\n"
        f"{rules_block}"
        f"HISTORICAL CONTEXT:\n{context_text}\n\n"
        "Draft the quote now."
    )


# Serves a historical quote PDF from the quotes folder by filename
@app.get("/pdf/{filename}")
def serve_pdf(filename: str):
    pdf_path = QUOTES_FOLDER / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")
    return FileResponse(pdf_path, media_type="application/pdf")


# Accepts RFQ fields and returns the top 3 most similar historical quotes
@app.post("/retrieve")
def retrieve_endpoint(rfq: RFQInput):
    try:
        results = run_retrieval(
            facility_size=rfq.facility_size,
            cooling_load=rfq.cooling_load,
            config_type=rfq.config_type,
            location=rfq.location,
            redundancy_tier=rfq.redundancy_tier,
            timeline=rfq.timeline,
        )
        return {"results": results}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {error}")


# Pydantic model for the eval endpoint — RFQ fields plus the generated quote text
class EvalInput(RFQInput):
    quote_text: str


# Finds the first dollar amount on a line containing a keyword and returns it as a float
def find_price_by_keyword(text, keyword):
    for line in text.splitlines():
        if keyword.lower() in line.lower():
            match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", line)
            if match:
                amount = float(match.group(1).replace(",", ""))
                if amount > 0:
                    return amount
    return None


# Extracts materials, labour, overhead, pre-margin, and total from a quote text
def extract_pricing_fields(text):
    materials = find_price_by_keyword(text, "Materials Subtotal")
    labour = find_price_by_keyword(text, "Labour") or find_price_by_keyword(text, "Labor")
    overhead = find_price_by_keyword(text, "Overhead")
    pre_margin = find_price_by_keyword(text, "Pre-Margin") or find_price_by_keyword(text, "Pre-margin")
    total = find_price_by_keyword(text, "TOTAL QUOTED") or find_price_by_keyword(text, "Total Quoted")
    return materials, labour, overhead, pre_margin, total


# Returns True if actual is within the allowed tolerance of expected
def within_tolerance(actual, expected, tolerance):
    if not expected:
        return False
    return abs(actual - expected) / expected <= tolerance


# Runs the four eval checks and returns a list of result dicts
def run_eval_checks(quote_text, location):
    checks = []

    # Check 1: required sections
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in quote_text.lower()]
    if missing:
        checks.append({
            "name": "Section check",
            "status": "fail",
            "detail": f"Missing: {', '.join(missing)}",
        })
    else:
        checks.append({
            "name": "Section check",
            "status": "pass",
            "detail": f"All {len(REQUIRED_SECTIONS)} required sections present",
        })

    # Check 2: location referenced
    location_keyword = location.split(",")[0].strip()
    if location_keyword.lower() in quote_text.lower():
        checks.append({
            "name": "Location check",
            "status": "pass",
            "detail": f"'{location_keyword}' referenced in generated quote",
        })
    else:
        checks.append({
            "name": "Location check",
            "status": "fail",
            "detail": f"'{location_keyword}' not found in generated quote",
        })

    # Check 3: pricing formula
    materials, labour, overhead, pre_margin, total = extract_pricing_fields(quote_text)
    if not all([materials, labour, overhead, pre_margin, total]):
        missing_fields = [
            name for name, val in zip(
                ["materials", "labour", "overhead", "pre_margin", "total"],
                [materials, labour, overhead, pre_margin, total],
            ) if not val
        ]
        checks.append({
            "name": "Formula check",
            "status": "fail",
            "detail": f"Could not extract: {', '.join(missing_fields)}",
        })
    else:
        expected_overhead = (materials + labour) * 0.18
        expected_total = pre_margin / 0.78
        overhead_ok = within_tolerance(overhead, expected_overhead, FORMULA_TOLERANCE)
        total_ok = within_tolerance(total, expected_total, FORMULA_TOLERANCE)
        formula_status = "pass" if (overhead_ok and total_ok) else "fail"
        checks.append({
            "name": "Formula check",
            "status": formula_status,
            "detail": (
                f"Materials ${materials:,.0f} + Labour ${labour:,.0f} → "
                f"Overhead expected ${expected_overhead:,.0f} / actual ${overhead:,.0f} {'✓' if overhead_ok else '✗'}  |  "
                f"Total expected ${expected_total:,.0f} / actual ${total:,.0f} {'✓' if total_ok else '✗'}"
            ),
        })

    # Check 4: price vs historical reference
    try:
        historical_text = (EXTRACTED_FOLDER / HISTORICAL_REFERENCE_FILE).read_text(encoding="utf-8")
        _, _, _, _, historical_total = extract_pricing_fields(historical_text)
    except Exception:
        historical_total = None

    if not total:
        checks.append({"name": "Price comparison", "status": "skip", "detail": "Generated total not extractable"})
    elif not historical_total:
        checks.append({"name": "Price comparison", "status": "skip", "detail": "Historical reference not readable"})
    else:
        diff_pct = abs(total - historical_total) / historical_total * 100
        price_status = "pass" if within_tolerance(total, historical_total, PRICE_TOLERANCE) else "fail"
        checks.append({
            "name": "Price comparison",
            "status": price_status,
            "detail": (
                f"Generated ${total:,.0f} vs historical {HISTORICAL_REFERENCE_FILE} ${historical_total:,.0f} "
                f"— {diff_pct:.1f}% apart (threshold {int(PRICE_TOLERANCE*100)}%)"
            ),
        })

    return checks


# Accepts a generated quote plus RFQ fields and returns structured eval results
@app.post("/run-eval")
def run_eval_endpoint(input_data: EvalInput):
    try:
        checks = run_eval_checks(input_data.quote_text, input_data.location)
        passed = sum(1 for c in checks if c["status"] == "pass")
        total_checks = sum(1 for c in checks if c["status"] != "skip")
        return {"checks": checks, "passed": passed, "total": total_checks}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Eval failed: {error}")


# Pydantic model for the judge endpoint — RFQ fields, generated quote, and top matched document text
class JudgeInput(RFQInput):
    quote_text: str
    top_match_text: str


# Builds the LLM-as-judge prompt using the RFQ, reference sections, and generated quote
def build_judge_prompt(rfq, reference_sections, generated_quote):
    rfq_summary = (
        f"Facility size: {rfq.facility_size}\n"
        f"Cooling load: {rfq.cooling_load}\n"
        f"Configuration type: {rfq.config_type}\n"
        f"Location: {rfq.location}\n"
        f"Redundancy tier: {rfq.redundancy_tier}\n"
        f"Timeline: {rfq.timeline}"
    )

    return f"""You are a senior mechanical engineer at TMI Climate Solutions reviewing a draft quote generated by an AI system. Your job is to assess whether this draft is ready to send to a customer.

You will score the draft on five dimensions, then give an overall recommendation.

---

RFQ INPUTS:
{rfq_summary}

---

REFERENCE — Engineering notes and pricing from the closest historical TMI quote:
{reference_sections}

---

GENERATED DRAFT QUOTE:
{generated_quote}

---

Score the draft on each of the following five dimensions. Use a scale of 1 to 5, where:
1 = completely wrong or missing
2 = present but significantly inadequate
3 = acceptable but with notable gaps
4 = good with minor issues
5 = matches the standard of a senior TMI engineer

Dimensions to score:

1. Section completeness — Are all required content areas covered: equipment specification, bill of materials, pricing summary, lead time, commercial terms, and engineering notes? Award full marks even if headings differ, as long as the content is present.

2. Technical specificity — Are the equipment specifications (coil configuration, fan type, controls, filtration) appropriate and specific to the stated cooling load, configuration type, and redundancy tier? Generic or placeholder specs should be penalised.

3. Location awareness — Does the engineering notes section contain location-specific content relevant to {rfq.location}? This includes climate design conditions, local water quality, applicable codes, and any regional environmental factors. Penalise generic notes that could apply to any location.

4. Pricing coherence — Do the line items in the bill of materials and the pricing summary reflect the stated equipment scope? Are the figures plausible given the cooling load and reference pricing? Penalise if line items are vague, inconsistent with the spec, or if the arithmetic does not match the scope.

5. Style match — Does the draft read like a professional TMI quotation? Assess tone, structure, assumption-flagging in engineering notes, and the use of precise technical language. Penalise AI-sounding filler or vague qualifications.

Respond in the following JSON format only. Do not include any text outside the JSON block.

{{
  "scores": [
    {{"dimension": "Section completeness", "score": <1-5>, "reasoning": "<one sentence>"}},
    {{"dimension": "Technical specificity", "score": <1-5>, "reasoning": "<one sentence>"}},
    {{"dimension": "Location awareness", "score": <1-5>, "reasoning": "<one sentence>"}},
    {{"dimension": "Pricing coherence", "score": <1-5>, "reasoning": "<one sentence>"}},
    {{"dimension": "Style match", "score": <1-5>, "reasoning": "<one sentence>"}}
  ],
  "recommendation": "<APPROVE | REVISE | REJECT>",
  "recommendation_reasoning": "<one to two sentences explaining the overall verdict>"
}}

Use APPROVE if the draft is ready to send with at most minor copy edits.
Use REVISE if the draft has the right structure but requires meaningful corrections before sending.
Use REJECT if the draft has fundamental errors in pricing, scope, or technical content that make it unsuitable as a starting point."""


# Calls Claude to judge the generated quote and returns structured scores and a recommendation
def call_llm_judge(prompt):
    try:
        anthropic_client = anthropic.Anthropic()
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        # Strip markdown code fences if Claude wraps the JSON
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
        return json.loads(raw_text)
    except Exception as error:
        raise ValueError(f"Judge API call or JSON parse failed: {error}")


# Accepts RFQ fields, the generated quote, and the top matched document, then returns judge scores
@app.post("/run-judge")
def run_judge_endpoint(input_data: JudgeInput):
    try:
        reference_sections = extract_relevant_sections(input_data.top_match_text)
        prompt = build_judge_prompt(input_data, reference_sections, input_data.quote_text)
        result = call_llm_judge(prompt)
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Judge failed: {error}")


# Pydantic model for the log-correction endpoint
class CorrectionInput(BaseModel):
    section_name: str
    original_text: str
    corrected_text: str


# Appends a timestamped correction entry to corrections.json in the project root
@app.post("/log-correction")
def log_correction_endpoint(input_data: CorrectionInput):
    try:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "section_name": input_data.section_name,
            "original_text": input_data.original_text,
            "corrected_text": input_data.corrected_text,
        }
        existing = []
        if CORRECTIONS_FILE.exists():
            try:
                existing = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append(entry)
        CORRECTIONS_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "ok", "total_corrections": len(existing)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Log correction failed: {error}")


# Pydantic model for the assign-section endpoint
class AssignmentInput(BaseModel):
    section_name: str
    assignee: str
    timestamp: str


# Appends a timestamped assignment entry to assignments.json in the project root
@app.post("/assign-section")
def assign_section_endpoint(input_data: AssignmentInput):
    try:
        entry = {
            "section_name": input_data.section_name,
            "assignee": input_data.assignee,
            "timestamp": input_data.timestamp,
        }
        existing = []
        if ASSIGNMENTS_FILE.exists():
            try:
                existing = json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append(entry)
        ASSIGNMENTS_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "logged"}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Assign section failed: {error}")


# Pydantic model for the retrieve-rules endpoint
class RulesQueryInput(BaseModel):
    query_text: str


# Accepts a query string and returns the top matching engineering rules from the knowledge base
@app.post("/retrieve-rules")
def retrieve_rules_endpoint(input_data: RulesQueryInput):
    try:
        rules = run_retrieve_rules(input_data.query_text)
        return {"rules": rules}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Rules retrieval failed: {error}")


# Pydantic model for the regenerate-section endpoint
class RegenerateSectionInput(BaseModel):
    section_name: str
    rfq: RFQInput
    rules: list
    context: list


# Builds a focused prompt asking Claude to regenerate only one named section's body content
def build_section_prompt(section_name, rfq, rules, context_texts):
    rfq_summary = (
        f"Facility size: {rfq.facility_size}\n"
        f"Cooling load: {rfq.cooling_load}\n"
        f"Configuration type: {rfq.config_type}\n"
        f"Location: {rfq.location}\n"
        f"Redundancy tier: {rfq.redundancy_tier}\n"
        f"Timeline: {rfq.timeline}"
    )
    rules_block = ""
    if rules:
        rules_lines = "\n".join(f"- {rule}" for rule in rules)
        rules_block = f"\nENGINEERING RULES TO APPLY:\n{rules_lines}\n"
    context_blocks = []
    for index, doc_text in enumerate(context_texts, start=1):
        relevant = extract_relevant_sections(doc_text)
        context_blocks.append(f"--- Historical Quote {index} ---\n{relevant}")
    context_text = "\n\n".join(context_blocks)
    return (
        f"You are a quoting assistant for TMI Climate Solutions.\n\n"
        f"Regenerate only the '{section_name}' section of a quote for the following RFQ.\n"
        f"Return only the section body content — no section heading, no other sections.\n\n"
        f"RFQ:\n{rfq_summary}\n"
        f"{rules_block}\n"
        f"HISTORICAL CONTEXT:\n{context_text}\n\n"
        f"Write the '{section_name}' section body now."
    )


# Streams a single regenerated section body token by token using SSE
@app.post("/regenerate-section")
def regenerate_section_endpoint(input_data: RegenerateSectionInput):
    prompt = build_section_prompt(
        input_data.section_name,
        input_data.rfq,
        input_data.rules,
        input_data.context,
    )

    def stream_tokens():
        try:
            anthropic_client = anthropic.Anthropic()
            with anthropic_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'error': str(error)})}\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


# Streams the AI-generated quote token by token using SSE so the frontend can show real progress
@app.post("/generate-quote-stream")
def generate_quote_stream(input_data: GenerateQuoteInput):
    query_text = (
        f"{input_data.config_type} {input_data.cooling_load} {input_data.location} "
        f"{input_data.redundancy_tier}"
    )
    engineering_rules = run_retrieve_rules(query_text)
    prompt = build_generation_prompt(input_data, input_data.retrieved_documents, engineering_rules)

    def stream_tokens():
        try:
            yield f"data: {json.dumps({'rules_applied': engineering_rules})}\n\n"
            anthropic_client = anthropic.Anthropic()
            with anthropic_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'error': str(error)})}\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")
