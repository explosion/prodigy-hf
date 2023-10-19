<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# 🏘 Prodigy-ANN

![](images/approach.png)

This repository contains a Prodigy plugin for recipes that involve approximate nearest neighbor (ANN) techniques to fetch relevant subsets of the data to curate. To encode the text this library uses
[sentence-transformers](https://sbert.org) and it uses
[hnswlib](https://github.com/nmslib/hnswlib) as an index for these vectors.

You can install this plugin via `pip`. 

```
pip install "prodigy-ann @ git+https://github.com/explosion/prodigy-ann"
```

To learn more about this plugin, you can check the [Prodigy docs](https://prodi.gy/docs/plugins/#ann).

## Issues? 

Are you have trouble with this plugin? Let us know on our [support forum](https://support.prodi.gy/) and we'll get back to you! 