# Test fixtures

## `dummy.pdf`

- **Source:** https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf
- **Provenance:** W3C Web Accessibility Initiative test files.
- **Content:** single page, PDF 1.4, deflate-encoded, contains the text
  "Dummy PDF file".
- **Why this file:** smallest publicly-served PDF with normal text that
  validates the real `pdf-extract` → `PdfExtractCrate::extract` round
  trip in `tests/ingest_pdf_e2e.rs` and the extractor unit tests.

## `scanned.pdf`

- **Source:** generated for this repository with Pillow: five lines of
  Arial 44 pt rendered to a 1275×1650 grayscale image (US letter at
  150 dpi) and saved as a single-page PDF.
- **Content:** an image only, with no text layer. The rendered text reads
  "Pump Station Maintenance Log", "The north pump was replaced on March 12.",
  "Inspector Mira Okafor signed off the repair." and "Next inspection is due
  in September."
- **Why this file:** a page with no extractable text forces the tiered
  extractor onto its OCR tier, so `ingest_pdf_tiered_test.rs` checks OCR
  recognition, page/block provenance and recall, not just the native tier.

## `scanned-page.jpg`

- **Source:** the page image embedded in `scanned.pdf` (its `DCTDecode`
  stream, written out unchanged): 1275×1650 grayscale JPEG.
- **Content:** the same five lines as `scanned.pdf`.
- **Why this file:** an image ingested on its own, so the model-backed OCR
  extractor (`ModelExtractor::image().ocr(..)`, issue #61) is tested on the
  same text the PDF OCR tier reads.
