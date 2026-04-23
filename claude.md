# Project: TMI quoting assistant -- RAG pipeline

## What this project is
A demo AI quoting system for TMI Climate Solutions, a company that builds custom air handling and cooling systems for data centers. The system reads incoming RFQ documents, retrieves relevant historical quotes from a vector database, and generates a structured quote output that feeds into their existing CPQ system.

## Project structure
- `quotes/` -- 30 synthetic historical quote PDFs, one per document (input)
- `extracted/` -- extracted text files, one per quote PDF (generated)
- `pipeline/` -- RAG pipeline code
- `demo/` -- demo interface code to be built after the pipeline

## Tech stack
Python for all backend code. ChromaDB for the vector database. Anthropic API for embeddings and reasoning. PDF extraction using pdfplumber.

## Coding principles -- follow these strictly
- Write simple, readable code.
- Every function should do one thing only.
- No function longer than 30 lines.
- Use clear variable names that describe what the variable contains -- no single letter variables, no abbreviations.
- Add a one line comment above every function explaining what it does.
- If something can be done in a simple way or a clever way, always choose the simple way.
- Code should be readable by someone who is not a Python expert.

## Key constraints
- Stay under 40 percent of model context window at all times.
- Retrieve maximum 3 documents per query.
- Similarity scores must be returned as percentages.
- Engineering notes section is the primary retrieval signal alongside location, cooling load, and configuration type.

## Pricing formula used in all quote documents
- Overhead = (materials + labour) x 18%
- Margin = pre-margin subtotal x 22/78
- Total = pre-margin subtotal / 0.78

## API key
Set as environment variable ANTHROPIC_API_KEY. Never hardcode.

## Error handling
Every function that reads a file or calls an API must have a try/except block with a clear error message that explains what went wrong and which file or call caused the problem.

## Before writing any code
- Always confirm the brief and outline the approach step by step before starting.
- Build and test one step at a time.
- Do not move to the next step until the current step is working and verified.
