# ocr-test

## ocr test repo

+ code examples for extracting text from PDFs using both pdfplumber and EasyOCR
+ main.py file for preprocessing


## Pdfplumber for regular PDFs and EasyOCR for scanned PDFs:



```python
import pdfplumber
import easyocr
import PIL
import numpy as np

def extract_text_from_pdf(pdf_path, use_ocr=False):
    if use_ocr:
        return extract_text_with_easyocr(pdf_path)
    else:
        return extract_text_with_pdfplumber(pdf_path)

def extract_text_with_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_with_easyocr(pdf_path):
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English language
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    
    full_text = ""
    for img in images:
        result = reader.readtext(np.array(img))
        text = " ".join([res[1] for res in result])
        full_text += text + "\n"
    
    return full_text

def pdf_to_images(pdf_path):
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    return [page.get_pixmap().tobytes() for page in doc]

# Usage example
pdf_path = "path/to/your/pdf/file.pdf"
is_scanned = False  # Set to True if dealing with scanned PDFs

extracted_text = extract_text_from_pdf(pdf_path, use_ocr=is_scanned)
print(extracted_text)

```

integrate this with the NLP extraction pipeline we created earlier:

```python
import spacy
import re
import dateparser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Include the PDF extraction functions here (extract_text_from_pdf, etc.)
# ...

def nlp_extraction_pipeline(text):
    # Preprocessing
    preprocessed_text = preprocess_text(text)
    
    # NLP processing
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Named Entity Recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Date extraction
    dates = extract_dates(text)
    
    # Custom extraction (e.g., email addresses)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = extract_custom_data(text, email_pattern)
    
    return {
        'preprocessed_text': preprocessed_text,
        'entities': entities,
        'dates': dates,
        'emails': emails
    }

# Helper functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def extract_dates(text):
    date_patterns = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}|\d{4})\b'
    dates = re.findall(date_patterns, text)
    return [dateparser.parse(date) for date in dates if dateparser.parse(date)]

def extract_custom_data(text, pattern):
    return re.findall(pattern, text)

# Main function to process PDF and extract information
def process_pdf(pdf_path, is_scanned=False):
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path, use_ocr=is_scanned)
    
    # Process extracted text with NLP pipeline
    extracted_data = nlp_extraction_pipeline(extracted_text)
    
    return extracted_data

# Usage
pdf_path = "path/to/your/pdf/file.pdf"
is_scanned = False  # Set to True if dealing with scanned PDFs

result = process_pdf(pdf_path, is_scanned)
print(result)

```

This integrated pipeline does the following:

1. Extracts text from a PDF file using either pdfplumber or EasyOCR, depending on whether the PDF is scanned or not.
2. Processes the extracted text using our NLP pipeline, which includes:
   - Text preprocessing
   - Named Entity Recognition
   - Date extraction
   - Custom data extraction (e.g., email addresses)

To use this pipeline:

1. Install the required libraries: `pip install pdfplumber easyocr spacy nltk dateparser PyMuPDF`
2. Download the necessary NLTK data: 
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
3. Download the spaCy model: `python -m spacy download en_core_web_sm`
4. Run the `process_pdf` function with the path to your PDF file.

You can easily extend this pipeline to extract other specific types of data by adding more custom extraction functions or by training a machine learning model for more complex extraction tasks.



## Goals

1. Text Preprocessing
2. NLP Pipeline Setup
3. Named Entity Recognition (NER)
4. Information Extraction
5. Custom Rule-Based Extraction

## Test

Let's go through each step:

1. Text Preprocessing:
First, you'll want to clean and normalize your extracted text.



```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Use this function on your extracted text
preprocessed_text = preprocess_text(your_extracted_text)
```

2. NLP Pipeline Setup:
For more advanced NLP tasks, you can use libraries like spaCy or Stanford NLP. Let's use spaCy for this example:

```python
import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Process the text
doc = nlp(your_extracted_text)
```

3. Named Entity Recognition (NER):
NER can help identify specific types of information in your text:

```python
for ent in doc.ents:
    print(f"Entity: {ent.text}, Type: {ent.label_}")
```

4. Information Extraction:
Depending on the specific data you need, you might use different techniques. Here's an example of extracting dates:

```python
import dateparser

def extract_dates(text):
    date_patterns = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}|\d{4})\b'
    dates = re.findall(date_patterns, text)
    return [dateparser.parse(date) for date in dates if dateparser.parse(date)]

extracted_dates = extract_dates(your_extracted_text)
```

5. Custom Rule-Based Extraction:
For specific data that follows certain patterns, you can create custom extraction rules:

```python
def extract_custom_data(text, pattern):
    matches = re.findall(pattern, text)
    return matches

# Example: Extract email addresses
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = extract_custom_data(your_extracted_text, email_pattern)
```


## main.py

Install the necessary libraries (spacy, nltk, dateparser) and download the required models (spacy's en_core_web_sm and nltk's punkt and stopwords).

PDF extraction:

1. Extract text from your PDF using pdfplumber or EasyOCR as you've been doing.
2. Pass the extracted text to the `nlp_extraction_pipeline` function.
3. Analyze the returned dictionary for the information you need.



```python
import spacy
import re
import dateparser
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def nlp_extraction_pipeline(text):
    # Preprocessing
    preprocessed_text = preprocess_text(text)
    
    # NLP processing
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Named Entity Recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Date extraction
    dates = extract_dates(text)
    
    # Custom extraction (e.g., email addresses)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = extract_custom_data(text, email_pattern)
    
    return {
        'preprocessed_text': preprocessed_text,
        'entities': entities,
        'dates': dates,
        'emails': emails
    }

# Helper functions (as defined earlier)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def extract_dates(text):
    date_patterns = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}|\d{4})\b'
    dates = re.findall(date_patterns, text)
    return [dateparser.parse(date) for date in dates if dateparser.parse(date)]

def extract_custom_data(text, pattern):
    matches = re.findall(pattern, text)
    return matches

# Usage
extracted_data = nlp_extraction_pipeline(your_extracted_text)
print(extracted_data)
```


