from transformers import pipeline
import spacy 
from spacy.tokens import Span

nlp = spacy.blank("en")
tfm_model = pipeline("ner", model="dslim/bert-base-NER")

text = "My name is Vincent D. Warmerdam. I work for the Dutch government."

class EntityMerger:
    """Helper class to make merging of B- I- tokens from hf-transformer easier."""
    def __init__(self, text, label, start, end):
        self.text = text
        self.label = label.replace("B-", "")
        self.start = start
        self.end = end
    
    def __repr__(self) -> str:
        return f"<Entity {self.label} {self.text[self.start:self.end]}>"

    def append_hf_tok(self, tok):
        assert tok['entity'].startswith("I-")
        assert tok['entity'].replace("I-", "") == self.label
        self.end = tok['end']

def to_spacy_doc(text, hf_model, nlp):
    entities = []
    current = None
    for ex in hf_model(text):
        if ex['entity'][0] == 'B':
            if current: 
                entities.append(current)
            current = EntityMerger(text=text, label=ex['entity'], start=ex['start'], end=ex['end'])
        else:
            current.append_hf_tok(ex)
    entities.append(current)

    doc = nlp(text)
    spans = []
    for ent in entities:
        span = doc.char_span(ent.start, ent.end, label=ent.label)
        spans.append(span)
    doc.ents = spans
    return doc

to_spacy_doc(text)