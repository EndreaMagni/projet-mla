PROJET MLA group 10 : <br>
===
# Description
This project is an implementation of the paper ["Neural machine translation by jointly learning to align and translate"][2] by Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio.

# Contributors
- Linda Rahoui 
- Feriel Bouchakour
- Raphaël Khorassani
- Wang Jing

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

| **Task**                                     | **Responsibles**                                 |
|---------------------------------------------|--------------------------------------------------|
| **Dataset :**                                                                                 |
| - Download the dataset and tokenize with Moses tokenizer | Feriel                                |
| - Retrieve the most frequent words and pad the dataset | Raphaël and Feriel           |
| **src Folder :**                                                                               |
| - main.py file                               | Everyone                                        |
| - Run.ipynb file                             | Raphaël                                          |
| **Baseline Folder :**                                                                          |
| - Display alignment model weights            | Wang                                             |
| - Script for training translation.py         | Everyone                                        |
| - RNNEncDec model architecture               | Raphaël                                          |
| **RNNSearch Model Architecture :**                                                             |
| - Encoder                                    | Feriel                                           |
| - Decoder                                    | Linda                                            |
| - Alignment model                            | Wang                                             |
| - Seq2Seq                                    | Feriel, Linda, Wang                              |
| **Evaluation Folder :**                                                                        |
| - Script to launch evaluation with BLEU score | Raphaël                                         |
| **GenerateData Folder :**                                                                      |
| - Dataloader script                          | Feriel                                           |



[2]: https://arxiv.org/abs/1409.0473 "lien article"
[3]: https://www.statmt.org/wmt14/translation-task.html "lien dataset"
