
# Working Memory Graphs
This repository accompanies 
the ICML 2020 paper [Working Memory Graphs](https://arxiv.org/abs/1911.07141)
by Ricky Loynd, Roland Fernandez, Asli Celikyilmaz, Adith Swaminathan and Matthew Hausknecht.

WMG is a Transformer-based RL agent that attends to
a dynamic set of vectors representing observed and recurrent state.

![](images/overview.png)

## Installation steps
(The code has been tested with Python 3.6, on both Windows and Linux.)
* Clone this repository.
* Create a new virtual environment.
* Activate the virtual environment.
* Install PyTorch 1.3.1.
* Install other dependencies:
	* cd wmg_agent
    * pip install -r requirements.txt
* Install BabyAI (from any directory):
	* conda install -c conda-forge python-blosc
	* git clone -b iclr19 https://github.com/mila-iqia/babyai.git
	* cd babyai
	* pip install --user --editable .


# Running experiments

Execute all run commands from the **wmg_agent** directory, using this format:

    python run.py <runspec>

Use the runspecs in the [spec](specs) directory to reproduce results from the paper,
and as examples to create new runspecs for new experiments.

# Citations

If using this code in your work, please cite as follows:

    @misc{wmg_agent_2020,
      author = {Ricky Loynd and Roland Fernandez and Asli Celikyilmaz and Adith Swaminathan and Matthew Hausknecht},
      title = {Working Memory Graphs: Source code},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/microsoft/wmg_agent}}
    }

This repo's implementation of Sokoban was derived in part from that of 
[https://github.com/mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban).
If using the Sokoban environment in your work, please honor that source's 
[license](environments/gym-sokoban-LICENSE), and cite as follows:

    @misc{SchraderSokoban2018,
      author = {Schrader, Max-Philipp B.},
      title = {gym-sokoban},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/mpSchrader/gym-sokoban}},
      commit = {#CommitId}
    }

For convenience, we provide some of the predefined Boxoban levels from [https://github.com/deepmind/boxoban-levels](https://github.com/deepmind/boxoban-levels).
If using this data in your work, please honor that source's 
[license](data/boxoban-levels-master/LICENSE), and cite as follows:

    @misc{boxobanlevels,
      author = {Arthur Guez, Mehdi Mirza, Karol Gregor, Rishabh Kabra, Sebastien Racaniere, Theophane Weber, David Raposo, Adam Santoro, Laurent Orseau, Tom Eccles, Greg Wayne, David Silver, Timothy Lillicrap, Victor Valdes},
      title = {An investigation of Model-free planning: boxoban levels},
      howpublished= {https://github.com/deepmind/boxoban-levels/},
      year = "2018",
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
