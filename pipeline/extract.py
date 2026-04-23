import pdfplumber
import pathlib

QUOTES_FOLDER = pathlib.Path("quotes")
EXTRACTED_FOLDER = pathlib.Path("extracted")
REQUIRED_SECTION = "ENGINEERING NOTES"


# Creates the output folder if it does not already exist
def create_output_folder(folder_path):
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        print(f"Error: could not create output folder '{folder_path}': {error}")
        raise


# Extracts all text from a PDF file by combining text from every page
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_pages_text.append(page_text)
            return "\n".join(all_pages_text)
    except Exception as error:
        print(f"Error: could not extract text from '{pdf_path.name}': {error}")
        return None


# Checks that the required section heading appears in the extracted text
def validate_engineering_notes(extracted_text, filename):
    if REQUIRED_SECTION not in extracted_text:
        print(f"Warning: '{REQUIRED_SECTION}' section not found in '{filename}' -- skipping")
        return False
    return True


# Saves extracted text to a .txt file in the output folder
def save_text_to_file(text, output_path):
    try:
        output_path.write_text(text, encoding="utf-8")
    except Exception as error:
        print(f"Error: could not save file '{output_path.name}': {error}")
        raise


# Processes a single PDF: extracts, validates, and saves the text
def process_single_pdf(pdf_path, output_folder):
    extracted_text = extract_text_from_pdf(pdf_path)
    if extracted_text is None:
        return False

    if not validate_engineering_notes(extracted_text, pdf_path.name):
        return False

    output_filename = pdf_path.stem + ".txt"
    output_path = output_folder / output_filename
    save_text_to_file(extracted_text, output_path)
    return True


# Loops through all PDFs in the quotes folder and processes each one
def process_all_pdfs(quotes_folder, output_folder):
    pdf_files = sorted(quotes_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{quotes_folder}'")
        return

    success_count = 0
    failure_count = 0

    for pdf_path in pdf_files:
        succeeded = process_single_pdf(pdf_path, output_folder)
        if succeeded:
            success_count += 1
        else:
            failure_count += 1

    print(f"\nSummary: {success_count} documents extracted successfully, {failure_count} failed validation.")


# Entry point: sets up folders and runs the extraction pipeline
def main():
    create_output_folder(EXTRACTED_FOLDER)
    process_all_pdfs(QUOTES_FOLDER, EXTRACTED_FOLDER)


if __name__ == "__main__":
    main()
