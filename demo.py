from transformers import pipeline
import spacy 
from spacy.tokens import Span

nlp = spacy.blank("en")
tfm_model = pipeline("ner", model="dslim/bert-base-NER")

text = "My name is Vincent D. Warmerdam. I work for the Dutch government."

class Entity:
    def __init__(self, text, label, start, end):
        self.text = text
        self.label = label.replace("B-", "")
        self.start = start
        self.end = end
    
    def __repr__(self) -> str:
        return f"<Entity {self.label} {self.text[self.start:self.end]}>"

    def append_hf_tok(self, tok):
        print(tok, type(tok))
        assert tok['entity'].startswith("I-")
        assert tok['entity'].replace("I-", "") == self.label
        self.end = tok['end']

def to_spacy_doc(text):
    entities = []
    current = None
    for ex in tfm_model(text):
        if ex['entity'][0] == 'B':
            if current: 
                entities.append(current)
            current = Entity(text=text, label=ex['entity'], start=ex['start'], end=ex['end'])
        else:
            current.append_hf_tok(ex)
    entities.append(current)
    print(entities)

    doc = nlp(text)
    spans = []
    for ent in entities:
        span = doc.char_span(ent.start, ent.end, label=ent.label)
        spans.append(span)
    doc.ents = spans
    print(doc.ents)

to_spacy_doc(text)