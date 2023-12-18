PROJET MLA group 10 : <br>
===
# Description
This project is an implementation of the paper ["Neural machine translation by jointly learning to align and translate"][2] by Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio.

# Contributors
Linda Rahoui 
Feriel Bouchakour
Raphaël Khorassani
Wang Jing

# Installation and Setup
To use this project, clone the repository and install the required libraries listed in requirements.txt.

# Usage
Run the provided [notebook](src/Run.ipynb) to execute the translation model. The notebook includes detailed instructions and examples.


# Dataset
The dataset used in this project can be found [here][3] . It includes parallel corpora for the languages involved in the translation model.

# Methodology
Our approach uses a sequence-to-sequence model with attention mechanisms to effectively translate while maintaining context.

# Task distribution
Here is the distribution of the taks we made :

| Tâches                                      | Responsables                                     |
|---------------------------------------------|--------------------------------------------------|
| **Dataset :**                               |                                                  |
| Télécharger le dataset et tokeniser avec Moses tokenizer | Feriel                                        |
| Récupérer les mots les plus fréquents et padder le dataset | Raphaël et Feriel                               |
| **Dossier src :**                            |                                                  |
| Fichier main.py                              | Tout le monde                                    |
| Fichier Run.ipynb                            | Raphaël                                          |
| **Dossier Baseline :**                       |                                                  |
| Affichage des poids de l’alignement model    | Wang                                             |
| Script pour l’entraînement translation.py   | Tout le monde                                    |
| Architecture modèle RNNEncDec               | Raphaël                                          |
| **Architecture modèle RNNSearch :**          |                                                  |
| Encodeur                                     | Feriel                                           |
| Décodeur                                     | Linda                                            |
| Alignement model                             | Wang                                             |
| Seq2Seq                                     | Feriel, Linda, Wang                              |
| **Dossier Evaluation :**                     |                                                  |
| Script pour lancer l’évaluation avec le BLEU score | Raphaël                                       |
| **Dossier GenerateData :**                   |                                                  |
| Script du dataloader                         | Feriel                                           |


[2]: https://arxiv.org/abs/1409.0473 "lien article"
[3]: https://www.statmt.org/wmt14/translation-task.html "lien dataset"
