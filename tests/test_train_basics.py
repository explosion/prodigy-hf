"""
These tests assume some datasets are available in the Prodigy database.
Check the `.github/workflows/tests.yml` file for more details.
"""

import pytest 
from pathlib import Path
from prodigy_hf import hf_train_ner, hf_train_textcat, hf_ner_correct, hf_textcat_correct


def test_smoke_ner(tmpdir):
    # Make sure we can train without errors
    hf_train_ner("fashion,eval:fashion", tmpdir, epochs=1, model_name="hf-internal-testing/tiny-random-DistilBertModel")
    
    # Make sure we throw this one error
    with pytest.raises(ValueError):
        hf_train_ner("does-not-exist", tmpdir, epochs=1, model_name="hf-internal-testing/tiny-random-DistilBertModel")
    
    # Make sure we can ner.correct the trained model without smoke 
    prodigy_dict = hf_ner_correct("xxx", f"{tmpdir}/checkpoint-2", "dataset:fashion")
    for ex in prodigy_dict['stream']:
        for span in ex['spans']:
            assert span['label'] == 'FASHION_BRAND'


@pytest.mark.parametrize("dataset", ["textcat-binary", "textcat-multi"])
def test_smoke_textcat(dataset, tmpdir):
    # Make sure we can train without errors
    hf_train_textcat(f"{dataset},eval:{dataset}", tmpdir, epochs=1, model_name="hf-internal-testing/tiny-random-DistilBertModel")

    # Make sure we throw this one error
    with pytest.raises(ValueError):
        hf_train_textcat("does-not-exist", tmpdir, epochs=1, model_name="hf-internal-testing/tiny-random-DistilBertModel")
    
    # Make sure we can ner.correct the trained model without smoke 
    prodigy_dict = hf_textcat_correct("xxx", f"{tmpdir}/checkpoint-2", "dataset:fashion")
    for ex in prodigy_dict['stream']:
        if "binary" in dataset:
            print(ex)
            assert ex['label'] == 'foo'
        else:
            assert set([lab['id'] for lab in ex['options']]) == set(["foo", "bar", "buz"])
