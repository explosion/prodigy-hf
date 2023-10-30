import pytest 
from prodigy_hf import hf_train_ner, hf_train_textcat


def test_smoke_ner(tmpdir):
    hf_train_ner("fashion,eval:fashion", tmpdir, epochs=1, model_name="hf-internal-testing/tiny-random-DistilBertModel")


@pytest.mark.parametrize("dataset", ["textcat-binary", "textcat-multi"])
def test_smoke_textcat(dataset, tmpdir):
    hf_train_textcat(f"{dataset},eval:{dataset}", tmpdir, epochs=1, model_name="hf-internal-testing/tiny-random-DistilBertModel")
