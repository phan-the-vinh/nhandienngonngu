# SUBTLEX-US Dictionary Dataset Notice & Attribution

## 1. Dataset Information
- **Dataset Name**: SUBTLEX-US (American English Word Frequency Norms)
- **Original Authors**: Marc Brysbaert (Ghent University) & Boris New (Université Savoie Mont Blanc / CNRS)
- **Source Publisher**: Department of Experimental Psychology, Ghent University (UGent), Belgium & OpenLexicon Project
- **Direct Source URL**: `https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip`
- **Original Archive File**: `subtlexus2.zip` (SHA-256: `67e595da1b399d2a21e25a1466a8d9f242f21a219c6bbbd3e089c4779ad83856`)
- **Original Uncompressed File**: `SUBTLEXus74286wordstextversion.txt` (SHA-256: `c5f86f065fc5d057fbf366433b8c5ca550aa7c24e128362dea4394f2b29c86e4`)
- **Download Timestamp**: 2026-11-07T13:50:50Z

## 2. Academic Citation
Any publication, research, or product utilizing this dataset should cite the primary foundational paper:

> **Brysbaert, M., & New, B. (2009).** Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. *Behavior Research Methods*, 41(4), 977–990. DOI: [10.3758/BRM.41.4.977](https://doi.org/10.3758/BRM.41.4.977)

Additional Part-of-Speech and extension reference:
> **Brysbaert, M., New, B., & Keuleers, E. (2012).** Adding part-of-speech information to the SUBTLEX-US word frequencies. *Behavior Research Methods*, 44(4), 991–997. DOI: [10.3758/s13428-012-0190-4](https://doi.org/10.3758/s13428-012-0190-4)

## 3. License Terms
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **License URI**: `https://creativecommons.org/licenses/by-sa/4.0/`
- **Summary of License Rights**:
  - **Share**: Copy and redistribute the material in any medium or format.
  - **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.
  - **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
  - **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

## 4. Notice of Changes & Derivative Processing
The file `data/dictionary/english_words_25k.tsv` is a derivative work generated via `tools/ingest_subtlex.py` with the following transformations applied to the original 74,286-word corpus:
1. **Column Extraction**: Extracted column 1 (`Word`) and column 2 (`FREQcount`), omitting all other columns.
2. **Header Removal**: Stripped the initial column header line (`Word\tFREQcount...`).
3. **Alphabetical Filtering**: Retained exclusively tokens matching regex `^[A-Za-z]+$` (filtering out non-alphabetic tokens and contractions).
4. **Length and Single-Character Preservation**: Retained valid single-letter English words (`A`, `I`) and words up to 25 characters.
5. **Case Normalization**: Converted all words to uppercase (`.upper()`).
6. **Collision & Duplicate Handling**: Merged any case collisions by aggregating integer frequencies (`sum(frequencies)`).
7. **Deterministic Ranking**: Sorted entries by primary key `-FREQcount` (descending frequency) and secondary key `Word` (alphabetical ascending).
8. **Size Truncation**: Selected exactly the top 25,000 entries.

## 5. ShareAlike Obligation for Derivative Data
In accordance with the ShareAlike condition of CC BY-SA 4.0, the processed dictionary file `english_words_25k.tsv` is distributed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

Independent application source code in the repository (e.g. `app/suggestion/dictionary.py`, `run.py`, etc.) represents an independent computer program and is not a derivative database work under CC BY-SA.
