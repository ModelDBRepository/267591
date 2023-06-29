The listed files are associated with a paper entitled "Models of Purkinje cell dendritic tree selection during early cerebellar development" by Mizuki Kato and Erik De Schutter.

The following python scripts are implemented for the NeuroDevSim software (De Schutter 2022, https://github.com/CNS-OIST/NeuroDevSim), and set to run with 64 cores.

1. Initial_environment.py
The shared initial environment (cycle 1 to 70) for Purkinje cell models (except for S0C_noGranuleCell). The script initiates 6 out of 12 phases of Granule cell migrations and 5 cycles of dendritic growth of Purkinje cells. It takes about 20 minutes to run a single simulation.

20 database files for the initial environment with different seeds (1 to 20) are obtained by following command:
    for i in {1..20}; do python initial_environment.py $i; done

2. Scenario0A_growOne.py
This script simulates the growths of Purkinje cell dendrites ignoring dendritic selection processes by randomly choosing single tree out of five candidates at the very beginning of the simulation. Running a single simulation with the script takes about 3 hours.

3. Scenario0B_growAll.py
The script simulates the dendritic growths when growing all five candidate dendritic trees without the selection phase. Each simulation takes about 3.5 hours.

4. Scenario0C_noGranuleCell.py
The script simulates the dendritic growths with no granule cell as an environment. Thus, the code does not use the database generated from initial_environment.py. A single simulation takes about 16 hours.

5. Scenario1_fixedTimings.py
This script controls the dendritic retraction phase by triggering the retractions at two fixed simulation cycles: at cycle 90 and cycle 97. At cycle 90, 3 out of 5 trees with smaller numbers of synapses with parallel fibers retract.Then, 1 out of the rest 2 trees is selected as a primary tree based on the number of synapses at cycle 97. A single simulation of this scenario takes around 3 hours.

6. Scenario2_1_maturationBySize.py
The script performs the selection by choosing a winner tree by size of each dendrite. The first retractions are initiated when a total number of fronts (cylindrical agents constituting dendrites) in a cell reaches 300, and the cell retracts trees not been comprised of more than 20 fronts. Then, the cell chooses a tree with the largest numbers of fronts as its primary tree when a whole cell has 400 fronts. Each simulation takes around 3 hours.

7. Scenario2_2_maturationBySynapses.py
This script manages the selection by choosing winners based on number of synapses with parallel fibers. The first selection starts when a whole cell has 90 synapses in total from 5 trees, and the cell retracts trees with less than 30 synapses. The second selection starts when a cell gets 180 synapses in total, and choose a tree with the largest number of synapses as its primary tree. One simulation with this script takes about 3 hours.

8. Scenario3_networkActivity.py
This scenario coordinates the tree selection by total strengths of synaptic signals. The first retraction happens when signals summed for the whole cell reaches 500. Average signal for each cell is calculated and trees with lower signals than the average retracts. When the summed signal of a cell hits 1,200, the cell selects a tree with highest signals. Each simulation takes about 2.5 hours with the script.


