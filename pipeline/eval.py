import re
import sys
import json
import pathlib
import requests

API_BASE = "http://localhost:8000"
EXTRACTED_FOLDER = pathlib.Path("extracted")
FORMULA_TOLERANCE = 0.02  # allow 2% rounding error on formula checks
PRICE_TOLERANCE = 0.30    # generated quote total must be within 30% of historical reference

TEST_RFQ = {
    "facility_size": "5000 sqm",
    "cooling_load": "900 kW",
    "config_type": "Chilled water AHU",
    "location": "Dallas, TX",
    "redundancy_tier": "N+1",
    "timeline": "12 weeks",
}

HISTORICAL_REFERENCE_FILE = "TMI-Q-2022-011.txt"

REQUIRED_SECTIONS = [
    "EQUIPMENT SPECIFICATION",
    "BILL OF MATERIALS",
    "PRICING SUMMARY",
    "LEAD TIME",
    "COMMERCIAL TERMS",
    "ENGINEERING NOTES",
]


# Calls /retrieve and returns the top 3 matched documents
def call_retrieve(rfq):
    try:
        response = requests.post(f"{API_BASE}/retrieve", json=rfq, timeout=30)
        response.raise_for_status()
        return response.json()["results"]
    except Exception as error:
        print(f"Error: retrieval API call failed: {error}")
        sys.exit(1)


# Calls /generate-quote-stream and collects the full streamed response text
def call_generate_stream(rfq, retrieved_documents):
    payload = dict(rfq)
    payload["retrieved_documents"] = retrieved_documents
    try:
        response = requests.post(
            f"{API_BASE}/generate-quote-stream", json=payload, stream=True, timeout=180
        )
        response.raise_for_status()
    except Exception as error:
        print(f"Error: generate-quote-stream API call failed: {error}")
        sys.exit(1)

    full_text = ""
    buffer = ""
    for raw_chunk in response.iter_content(chunk_size=None):
        buffer += raw_chunk.decode("utf-8")
        lines = buffer.split("\n")
        buffer = lines.pop()
        for line in lines:
            if not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str == "[DONE]":
                return full_text
            try:
                parsed = json.loads(payload_str)
                if "chunk" in parsed:
                    full_text += parsed["chunk"]
            except json.JSONDecodeError:
                pass
    return full_text


# Reads the historical reference quote text from the extracted folder
def load_historical_reference(filename):
    file_path = EXTRACTED_FOLDER / filename
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as error:
        print(f"Error: could not read historical reference '{filename}': {error}")
        sys.exit(1)


# Finds the first dollar amount on a line and returns it as a float
def extract_dollar_amount_from_line(line):
    match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", line)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


# Searches a block of text for a line matching a keyword and returns the dollar amount found on it
def find_price_by_keyword(text, keyword):
    for line in text.splitlines():
        if keyword.lower() in line.lower():
            amount = extract_dollar_amount_from_line(line)
            if amount and amount > 0:
                return amount
    return None


# Extracts the four pricing fields needed for formula verification
def extract_pricing_fields(text):
    materials = find_price_by_keyword(text, "Materials Subtotal")
    labour = find_price_by_keyword(text, "Labour") or find_price_by_keyword(text, "Labor")
    overhead = find_price_by_keyword(text, "Overhead")
    pre_margin = find_price_by_keyword(text, "Pre-Margin") or find_price_by_keyword(text, "Pre-margin")
    total = find_price_by_keyword(text, "TOTAL QUOTED") or find_price_by_keyword(text, "Total Quoted")
    return materials, labour, overhead, pre_margin, total


# Checks whether a value is within the allowed tolerance of an expected value
def within_tolerance(actual, expected, tolerance):
    if expected == 0:
        return False
    return abs(actual - expected) / expected <= tolerance


# Runs all four eval checks and prints a report
def run_eval(generated_text, historical_text, rfq):
    results = []

    # --- Check 1: Required sections present ---
    missing_sections = [
        section for section in REQUIRED_SECTIONS
        if section.lower() not in generated_text.lower()
    ]
    if missing_sections:
        results.append(("FAIL", "Section check", f"Missing sections: {', '.join(missing_sections)}"))
    else:
        results.append(("PASS", "Section check", f"All {len(REQUIRED_SECTIONS)} required sections present"))

    # --- Check 2: Location mentioned ---
    location_keyword = rfq["location"].split(",")[0].strip()
    if location_keyword.lower() in generated_text.lower():
        results.append(("PASS", "Location check", f"'{location_keyword}' referenced in generated quote"))
    else:
        results.append(("FAIL", "Location check", f"'{location_keyword}' not found in generated quote"))

    # --- Check 3: Pricing formula ---
    materials, labour, overhead, pre_margin, total = extract_pricing_fields(generated_text)

    if not all([materials, labour, overhead, pre_margin, total]):
        missing = [
            name for name, val in zip(
                ["materials", "labour", "overhead", "pre_margin", "total"],
                [materials, labour, overhead, pre_margin, total]
            ) if not val
        ]
        results.append(("FAIL", "Formula check", f"Could not extract pricing fields: {', '.join(missing)}"))
    else:
        expected_overhead = (materials + labour) * 0.18
        expected_total = pre_margin / 0.78
        overhead_ok = within_tolerance(overhead, expected_overhead, FORMULA_TOLERANCE)
        total_ok = within_tolerance(total, expected_total, FORMULA_TOLERANCE)

        formula_lines = [
            f"  Materials: ${materials:,.2f}  Labour: ${labour:,.2f}",
            f"  Overhead expected: ${expected_overhead:,.2f}  actual: ${overhead:,.2f}  {'OK' if overhead_ok else 'MISMATCH'}",
            f"  Total expected: ${expected_total:,.2f}  actual: ${total:,.2f}  {'OK' if total_ok else 'MISMATCH'}",
        ]
        if overhead_ok and total_ok:
            results.append(("PASS", "Formula check", "\n".join(formula_lines)))
        else:
            results.append(("FAIL", "Formula check", "\n".join(formula_lines)))

    # --- Check 4: Price comparison against historical reference ---
    _, _, _, _, historical_total = extract_pricing_fields(historical_text)

    if not total:
        results.append(("SKIP", "Price comparison", "Generated total could not be extracted"))
    elif not historical_total:
        results.append(("SKIP", "Price comparison", f"Historical total could not be extracted from {HISTORICAL_REFERENCE_FILE}"))
    else:
        diff_pct = abs(total - historical_total) / historical_total * 100
        detail = (
            f"Generated: ${total:,.2f}  Historical ({HISTORICAL_REFERENCE_FILE}): ${historical_total:,.2f}  "
            f"Difference: {diff_pct:.1f}%"
        )
        if within_tolerance(total, historical_total, PRICE_TOLERANCE):
            results.append(("PASS", "Price comparison", detail))
        else:
            results.append(("FAIL", "Price comparison", detail))

    # --- Print report ---
    print("\n" + "=" * 60)
    print("EVAL REPORT")
    print("=" * 60)
    passed = sum(1 for r in results if r[0] == "PASS")
    total_checks = sum(1 for r in results if r[0] != "SKIP")
    print(f"Result: {passed}/{total_checks} checks passed\n")

    for status, name, detail in results:
        marker = "✓" if status == "PASS" else ("–" if status == "SKIP" else "✗")
        print(f"  {marker}  {name}")
        for line in detail.splitlines():
            print(f"       {line}")
        print()

    print("=" * 60)
    return passed == total_checks


# Entry point: runs the full pipeline then evaluates the generated quote
def main():
    print("Step 1/3  Retrieving similar quotes...")
    retrieved_documents = call_retrieve(TEST_RFQ)
    for doc in retrieved_documents:
        print(f"  {doc['similarity']}%  {doc['filename']}")

    print("\nStep 2/3  Generating quote (streaming)...")
    generated_text = call_generate_stream(TEST_RFQ, retrieved_documents)
    print(f"  Generated {len(generated_text)} characters")

    print("\nStep 3/3  Loading historical reference and running checks...")
    historical_text = load_historical_reference(HISTORICAL_REFERENCE_FILE)

    passed = run_eval(generated_text, historical_text, TEST_RFQ)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
