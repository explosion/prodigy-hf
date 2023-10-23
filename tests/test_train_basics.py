from prodigy_hf import hf_train_ner


def test_smoke_ner(tmpdir):
    hf_train_ner("fashion,eval:fashion", tmpdir, epochs=1, model_name="sshleifer/tiny-distilroberta-base")