
# Runspecs
A runspec defines all the settings for a single run.
To execute any of the runspecs in this directory (with **wmg_agent** as the current directory), type:

    python run.py specs/<runspec>

## BabyAI runspecs

## Pathfinding runspecs

## Sokoban runspecs

### test_sokoban.py

This loads **models/sokoban.pth** (one of the 20 WMG agent models trained on 20M environment interactions),
then tests the agent on the official [test set](../data/boxoban-levels-master/unfiltered/test) of 1000 puzzles.
The success rate should be around 84%.

### train_sokoban.py

This trains a new WMG agent on Sokoban puzzles drawn randomly from the [train set](../data/boxoban-levels-master/unfiltered/train),
which contains 10k puzzles.
In order to train an agent on the full train set of 900k puzzles, 
first copy all 900 files from [https://github.com/deepmind/boxoban-levels](https://github.com/deepmind/boxoban-levels) 
into the [train set directory](../data/boxoban-levels-master/unfiltered/train).

### render_sokoban.py

This loads **models/sokoban.pth**
and displays a series of Sokoban puzzles drawn randomly from the [train set](../data/boxoban-levels-master/unfiltered/train).
* To interact with the display: 
     * Use the arrow keys to choose the agent's move.
     * Press **space** to let the agent choose the next move.
     * Press **N** to skip to the next puzzle.
     * Press **Esc** to exit.
