## Morphology & generalisation

#### Installation

Install OpenNMT via the conventional route (e.g. https://pypi.org/project/OpenNMT-py/1.2.0/).
Afterwards, replace your implementation with the OpenNMT code provided here, and rerunning `python setup.py install`. 

#### Model training

This repository provides a custom adapted installation of OpenNMT in `opennmt`,
with a dedicated README on the scripts to train encoder-decoder models and run the various behavioural tests.

#### DC training and visualising results

Afterwards, the results presented in the paper can reproduced with code from the following folders:
- `behavioural`
  - `visualise_nonce.ipynb`: jupyter notebook to visualise nonce predictions.
  - `visualise_training_curve.ipynb`: jupyter notebook to visualise the training curves.
  - `visualise_overgeneralisation.ipynb`: jupyter notebook to visualise the overgeneralisation curves.
  - `visualise_enforce_gender.ipynb`: jupyter notebook to visualise the plural classes after enforcing gender.
  - `visualise_increasing_lengths.ipynb`: jupyter notebook to visualise the increasing lengths for the -s class.
- `diagnostic_classification`:
  - Contains various bash and python scripts to train DCs. Visit the folder for suggestions on how to train DCs and evaluate them.
  - Afterwards, `visualise_diagnostic_classification.ipynb` can help visualise the results,
  - and `baselines.ipynb` helps you collect baseline results.
- `interventions`:
  - Contains various bash and python scripts to perform interventions. Visit the folder for suggestions on how to train DCs and evaluate them.
  - Afterwards, `visualise_interventions.ipynb` can help visualise the results.
- `belth_model`: Decision trees of the models by Belth et al. (2021), retrained on Wiktionary data for 5 seeds. (https://arxiv.org/pdf/2105.05790.pdf)

The graphic below summarises the results per plural class, where the line thickness indicates relative performance, and colour gradients indicate how performance increases as a word is being processed.

<image width=650 src="schematic_overview_results.png" />

```
@inproceedings{dankers2021generalising,
  title={Generalising to German plural noun classes, from the perspective of a recurrent neural network},
  author={Dankers, Verna and Langedijk, Anna and McCurdy, Kate and Williams, Adina and Hupkes, Dieuwke},
  booktitle={Proceedings of the 25th Conference on Computational Natural Language Learning},
  pages={94--108},
  year={2021}
}
```
