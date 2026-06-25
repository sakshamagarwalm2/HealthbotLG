"""
HealthBot Test Suite
====================
Tests the search query builder, Tavily search, and summarization
with 20+ different inputs covering various edge cases.

Results are saved to: healthbot_test_results.txt
"""

import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# --- Setup -----------------------------------------------------------
load_dotenv('config.env', override=True)
assert os.getenv('OPENAI_API_KEY'), "Missing OPENAI_API_KEY"
assert os.getenv('TAVILY_API_KEY'), "Missing TAVILY_API_KEY"

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
tavily_tool = TavilySearchResults(max_results=5, include_answer=True, include_raw_content=True)

SEARCH_FAILURE_PHRASES = [
    "unable to retrieve", "no search results", "no results",
    "could not find", "i'm sorry", "i cannot", "no information",
    "was unable", "not able to", "no data", "nothing found",
]

# --- Core Functions (copied from notebook) --------------------------

def build_medical_query(user_input: str) -> str:
    prompt = [
        SystemMessage("You are a medical triage assistant. Determine if the input is health-related. "
                      "If YES: output a precise medical search query. "
                      "If NO: output 'REJECT:' followed by a short reason.\n\n"
                      "Examples:\n"
                      "- 'diabetes' -> diabetes causes symptoms treatment doctors\n"
                      "- 'chest pain' -> chest pain medical condition causes treatment\n"
                      "- 'how to fix a car' -> REJECT:not a health topic\n"
                      "- 'pasta recipe' -> REJECT:not a health topic\n"
                      "- 'heart' -> heart disease causes symptoms treatment doctors"),
        HumanMessage(f"Patient says: '{user_input}'")
    ]
    return llm.invoke(prompt).content.strip()


def is_failure_summary(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in SEARCH_FAILURE_PHRASES)


def run_search(query: str):
    """Run Tavily search and return raw results."""
    try:
        result = tavily_tool.invoke({"query": query})
        if result and len(result) > 0:
            return {"success": True, "count": len(result), "data": str(result)}
        return {"success": False, "count": 0, "data": "Empty results"}
    except Exception as e:
        return {"success": False, "count": 0, "data": f"Error: {e}"}


def run_summarize(search_text: str):
    """Run the summarization LLM call."""
    if not search_text or len(search_text.strip()) < 50:
        return {
            "summary": "SEARCH FAILED: Insufficient search results.",
            "success": False
        }

    system_prompt = SystemMessage(
        content=(
            "You are a patient education specialist at a hospital. "
            "Using ONLY the search results below, create a structured health guide.\n\n"
            "Cover ALL of these sections if the information is available:\n"
            "1. WHAT IS IT? - Brief description of the condition.\n"
            "2. CAUSES & RISK FACTORS - What causes it, who is at higher risk.\n"
            "3. COMMON SYMPTOMS - List symptoms a patient might experience.\n"
            "4. DIAGNOSIS & TESTS - How doctors diagnose it (exams, scans, labs).\n"
            "5. TREATMENT OPTIONS - Available treatments, medications, procedures.\n"
            "6. WHICH DOCTOR TO SEE - Type of specialist (e.g. cardiologist, neurologist).\n"
            "7. EMERGENCY WARNING SIGNS - When to seek immediate medical help.\n"
            "8. COMPLICATIONS - What can happen if left untreated.\n"
            "9. PREVENTION TIPS - Lifestyle changes to prevent or manage.\n"
            "10. PROGNOSIS & RECOVERY - Expected outlook, recovery time.\n\n"
            "Format each section with a clear header and 2-4 bullet points.\n"
            "Use plain, simple language. Define any medical terms.\n"
            "Be warm and reassuring.\n\n"
            f"Search results:\n\n{search_text}"
        )
    )
    response = llm.invoke([system_prompt])
    summary_text = response.content
    actually_successful = not is_failure_summary(summary_text) and len(summary_text.strip()) >= 50
    return {"summary": summary_text, "success": actually_successful}


# --- Test Cases ------------------------------------------------------

TEST_CASES = [
    # (id, category, user_input, description)
    # -- Common medical conditions --
    (1, "condition", "diabetes", "Common condition - exact name"),
    (2, "condition", "hypertension", "Common condition - exact name"),
    (3, "condition", "asthma", "Common condition - exact name"),
    (4, "condition", "type 2 diabetes", "Condition with type specification"),
    (5, "condition", "covid-19", "Infectious disease"),

    # -- Symptom descriptions --
    (6, "symptom", "I have chest pain and shortness of breath", "Multiple symptoms description"),
    (7, "symptom", "my head hurts and I feel dizzy", "Common symptom description"),
    (8, "symptom", "feeling tired all the time and losing weight", "Vague symptoms"),
    (9, "symptom", "legs are numb and can't stand properly", "Symptom from user's earlier test"),
    (10, "symptom", "sharp pain in lower right abdomen", "Specific symptom - possible appendicitis"),

    # -- Misspellings and typos --
    (11, "misspelling", "diabeties", "Misspelled condition"),
    (12, "misspelling", "hyPERTension", "Mixed case condition"),
    (13, "misspelling", "head ache", "Two-word misspelling"),
    (14, "misspelling", "cancer", "Short exact condition"),
    (15, "misspelling", "artheritis", "Misspelled condition"),

    # -- Vague / short inputs --
    (16, "vague", "heart", "Single word - could be ambiguous"),
    (17, "vague", "pain", "Single vague symptom"),
    (18, "vague", "sick", "Very vague"),
    (19, "vague", "a", "Single letter"),
    (20, "vague", "x", "Single character (noise)"),

    # -- Long and complex inputs --
    (21, "complex", "I am a 65 year old male with high blood pressure and diabetes. Recently I have been feeling chest tightness and my left arm hurts. What could be happening?", "Detailed elderly patient scenario"),
    (22, "complex", "my 5 year old son has high fever since 3 days, cough and runny nose", "Pediatric symptoms"),

    # -- Non-medical topics --
    (23, "non-medical", "how to fix a car", "Non-medical input"),
    (24, "non-medical", "recipe for pasta", "Non-medical input"),
]

# --- Results ---------------------------------------------------------

results = []
passed = 0
failed = 0
total = len(TEST_CASES)

print("=" * 70)
print(f"  HealthBot Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total test cases: {total}")
print("=" * 70)

for tid, category, user_input, desc in TEST_CASES:
    print(f"\n{'-' * 70}")
    print(f"  Test #{tid:2d} [{category:12s}] {desc}")
    print(f"  Input: \"{user_input}\"")
    print(f"{'-' * 70}")

    test_result = {
        "id": tid,
        "category": category,
        "description": desc,
        "user_input": user_input,
        "med_query": "",
        "search": {},
        "summary": {},
        "status": "",
        "errors": [],
    }

    # Step 1: Build medical query
    try:
        med_query = build_medical_query(user_input)
        test_result["med_query"] = med_query
        print(f"  |- Query: \"{med_query}\"")

        if med_query.startswith("REJECT:"):
            reason = med_query.replace("REJECT:", "").strip()
            print(f"  |- Result: SAFELY REJECTED (non-medical: {reason})")
            test_result["status"] = "PASS"
            test_result["search"] = {"success": False, "count": 0, "data": f"REJECTED: {reason}"}
            test_result["summary"] = {"success": True, "summary": f"REJECTED: {reason}"}
            results.append(test_result)
            passed += 1
            time.sleep(0.3)
            continue

        time.sleep(0.5)
    except Exception as e:
        test_result["med_query"] = f"ERROR: {e}"
        test_result["errors"].append(f"build_medical_query failed: {e}")
        print(f"  |- Query: ERROR - {e}")
        test_result["status"] = "FAIL"
        results.append(test_result)
        failed += 1
        continue

    # Step 2: Search
    try:
        search_result = run_search(med_query)
        test_result["search"] = search_result
        if search_result["success"]:
            print(f"  |- Search: {search_result['count']} results")
        else:
            print(f"  |- Search: FAILED - {search_result['data'][:80]}")
    except Exception as e:
        test_result["search"] = {"success": False, "data": str(e)}
        test_result["errors"].append(f"search failed: {e}")
        print(f"  |- Search: ERROR - {e}")
        test_result["status"] = "FAIL"
        results.append(test_result)
        failed += 1
        continue

    # Step 3: Summarize
    try:
        search_text = search_result.get("data", "")
        summary_result = run_summarize(search_text)
        test_result["summary"] = summary_result
        if summary_result["success"]:
            lines = summary_result["summary"].count('\n')
            print(f"  |- Summary: {len(summary_result['summary'])} chars, {lines} lines")
            test_result["status"] = "PASS"
            passed += 1
        else:
            print(f"  |- Summary: GENERATED BUT READS AS FAILURE")
            test_result["status"] = "PARTIAL"
            passed += 0.5
            failed += 0.5
    except Exception as e:
        test_result["summary"] = {"success": False, "data": str(e)}
        test_result["errors"].append(f"summarize failed: {e}")
        print(f"  |- Summary: ERROR - {e}")
        test_result["status"] = "FAIL"
        results.append(test_result)
        failed += 1
        continue

    results.append(test_result)
    time.sleep(0.3)  # Rate limiting

# --- Summary ---------------------------------------------------------
print(f"\n{'=' * 70}")
print(f"  RESULTS SUMMARY")
print(f"{'=' * 70}")
print(f"  Total:  {total}")
print(f"  Passed: {int(passed)}")
print(f"  Failed: {int(failed)}")
print(f"  Rate:   {passed/total*100:.1f}%")

# --- Save Results ----------------------------------------------------
output_lines = []
output_lines.append("=" * 80)
output_lines.append("  HealthBot Test Suite Results")
output_lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output_lines.append(f"  Total: {total}  |  Passed: {int(passed)}  |  Failed: {int(failed)}  |  Rate: {passed/total*100:.1f}%")
output_lines.append("=" * 80)

for r in results:
    output_lines.append("")
    output_lines.append("-" * 80)
    output_lines.append(f"  Test #{r['id']:2d} [{r['category']:12s}] {r['description']}")
    output_lines.append(f"  Status: {r['status']}")
    output_lines.append(f"  User Input: \"{r['user_input']}\"")
    output_lines.append(f"  Search Query: \"{r['med_query']}\"")
    output_lines.append(f"  Search Results: {r['search'].get('count', 'N/A') if r['search'].get('success') else 'FAILED'}")
    if r['errors']:
        for e in r['errors']:
            output_lines.append(f"  ERROR: {e}")
    if r.get('summary', {}).get('success'):
        summary = r['summary']['summary']
        output_lines.append(f"  Summary ({len(summary)} chars):")
        for line in summary.split('\n'):
            output_lines.append(f"    {line}")
    elif r.get('summary', {}).get('summary'):
        output_lines.append(f"  Summary (FAILURE): {r['summary']['summary'][:100]}")

with open('healthbot_test_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\n  Detailed results saved to: healthbot_test_results.txt")
print(f"{'=' * 70}")
