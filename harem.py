# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HAREM dataset"""


import json
import unicodedata
from typing import List, Tuple

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
@inproceedings{santos2006harem,
  title={Harem: An advanced ner evaluation contest for portuguese},
  author={Santos, Diana and Seco, Nuno and Cardoso, Nuno and Vilela, Rui},
  booktitle={quot; In Nicoletta Calzolari; Khalid Choukri; Aldo Gangemi; Bente Maegaard; Joseph Mariani; Jan Odjik; Daniel Tapias (ed) Proceedings of the 5 th International Conference on Language Resources and Evaluation (LREC'2006)(Genoa Italy 22-28 May 2006)},
  year={2006}
}
"""

_DESCRIPTION = """
The HAREM is a Portuguese language corpus commonly used for Named Entity Recognition tasks. It includes about 93k words, from 129 different texts,
from several genres, and language varieties. The split of this dataset version follows the division made by [1], where 7% HAREM
documents are the validation set and the miniHAREM corpus (with about 65k words) is the test set. There are two versions of the dataset set,
a version that has a total of 10 different named entity classes (Person, Organization, Location, Value, Date, Title, Thing, Event,
Abstraction, and Other) and a "selective" version with only 5 classes (Person, Organization, Location, Value, and Date).
It's important to note that the original version of the HAREM dataset has 2 levels of NER details, namely "Category" and "Sub-type".
The dataset version processed here ONLY USE the "Category" level of the original dataset.
[1] Souza, Fábio, Rodrigo Nogueira, and Roberto Lotufo. "BERTimbau: Pretrained BERT Models for Brazilian Portuguese." Brazilian Conference on Intelligent Systems. Springer, Cham, 2020.
"""

_HOMEPAGE = "https://www.linguateca.pt/primeiroHAREM/harem_coleccaodourada_en.html"

_LICENSE = ""

_URL = "https://github.com/eduagarcia/harem/raw/master/"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"

class HAREM(datasets.GeneratorBasedBuilder):
    """HAREM dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="total",
            version=VERSION,
            description="All the tags (PESSOA, ORGANIZACAO, LOCAL, TEMPO, VALOR, ABSTRACCAO, ACONTECIMENTO, COISA, OBRA, OUTRO) will be used",
        ),
        datasets.BuilderConfig(
            name="selective",
            version=VERSION,
            description="Only a subset of the tags (PESSOA, ORGANIZACAO, LOCAL, TEMPO, VALOR) will be used",
        ),
    ]

    DEFAULT_CONFIG_NAME = "total"

    def _info(self):

        tags = [
            "O",
            "B-PESSOA",
            "I-PESSOA",
            "B-ORGANIZACAO",
            "I-ORGANIZACAO",
            "B-LOCAL",
            "I-LOCAL",
            "B-TEMPO",
            "I-TEMPO",
            "B-VALOR",
            "I-VALOR",
        ]

        if self.config.name == "total":
            tags += [
                "B-ABSTRACCAO",
                "I-ABSTRACCAO",
                "B-ACONTECIMENTO",
                "I-ACONTECIMENTO",
                "B-COISA",
                "I-COISA",
                "B-OBRA",
                "I-OBRA",
                "B-OUTRO",
                "I-OUTRO",
            ]

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=tags)),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{self.config.name}/{_TRAINING_FILE}",
            "dev": f"{_URL}{self.config.name}/{_DEV_FILE}",
            "test": f"{_URL}{self.config.name}/{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""

        logger.info("⏳ Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[3].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }