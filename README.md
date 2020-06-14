
# Working Memory Graphs
This repository (wmg_agent) is the open source release associated with 
the ICML 2020 paper [Working Memory Graphs](https://arxiv.org/abs/1911.07141).
Use the following steps to reproduce key results from that work.


## Installation Steps
* Clone this repository.
	<br/><br/>
* Create a new virtual environment (for python 3.6).
	<br/><br/>
* Activate the environment.
	<br/><br/>
* Install PyTorch 1.3.1:
	* **conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch**
	<br/><br/>
* Install dependencies:
	* **cd wmg_agent**
    * **pip install -r requirements.txt**
	<br/><br/>

# To run

To run a trained Sokoban agent with display:

    At the wmg_agent directory, type:
        python scripts\rl_display.py --load_model models\sokoban.pth
    
    Then:
        Enter moves with the arrow keys.
        Press space to let agent move.
        Press N to skip to the next puzzle.
        Press Esc to exit.

## (More documentation coming)

# Citing

If using this code for your research, please cite as follows:

    @misc{wmg_agent_2020,
      author = {Loynd, Ricky},
      title = {wmg_agent},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/microsoft/wmg_agent}},
      commit = {#CommitId}
    }

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
