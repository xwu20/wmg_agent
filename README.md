
# Working Memory Graphs
This repository is provided to assist in reproducing results from 
the paper [Working Memory Graphs](https://arxiv.org/abs/1911.07141)
published at ICML 2020.


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

All runs should be executed with wmg_agent as the current directory.

All runs follow this format:

    python run.py <runspec>

The runspecs in the [spec](specs) directory may be used as-is,
or new specs may be created manually.

To run a trained Sokoban agent with display:

    python run.py specs/render_sokoban.py
    
    Then:
        Enter moves with the arrow keys.
        Press space to let agent move.
        Press N to skip to the next puzzle.
        Press Esc to exit.

To train a Sokoban agent:

    python run.py specs/train_sokoban.py
    
### (More documentation coming)

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

Our implementation of Sokoban was derived in part from [https://github.com/mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban).
If using the Sokoban environment in your work, please honor their [license](environments/gym-sokoban-LICENSE), and cite as follows:

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
If using this data in your work, please honor their [license](data/boxoban-levels-master/LICENSE), and cite as follows:

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
