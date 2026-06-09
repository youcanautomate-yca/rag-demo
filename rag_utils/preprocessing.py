import re
def preprocess(text: str):
    # Basic preprocessing: stripping whitespace and normalizing spaces
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    cleaned_text = text.strip()
    return cleaned_text
