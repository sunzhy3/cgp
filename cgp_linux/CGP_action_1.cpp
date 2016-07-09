#include "stdafx.h"
#include "cgp_mat.h"
#include "node_function.h"

int main()
{
	struct parameters* params = NULL;
	struct dataSet* training_data = NULL;
	struct chromosome* chromo = NULL;

	int num_inputs = 8;
	int num_nodes = 50;
	int num_outputs = 8;
	int node_arity = 2;

	//allocate for training data 
	Mat *inputs = new Mat[(const int) num_inputs];
	Mat *outputs = new Mat[(const int) num_outputs];

	string file_path = "/home/szy/cgp_linux/image/v_shooting_01_01/training_data";

	getTrainingData(file_path, inputs, outputs, num_inputs, num_outputs);

	int num_gens = 100000;
	double mutation_rate = 0.1;
	double target_fitness = 0.1;
	int update_frequency = 200;
	int num_threads = 4;

	params = initialiseParameters(num_inputs, num_nodes, num_outputs, node_arity);
	
	setCustomFitnessFunction(params, supervisedLearningIU, "supervisedLearningIU");

	setMutationRate(params, mutation_rate);
	
	setNumThreads(params, num_threads);

	setTargetFitness(params, target_fitness);

	setUpdateFrequency(params, update_frequency);

	printParameters(params);

	training_data = initialiseDataSetFromArrays(num_inputs, num_outputs, 1, inputs, outputs);

	chromo = runCGP(params, training_data, num_gens);

	printChromosome(chromo, 0);

	saveChromosome(chromo, "best_result.chromo");

	executeChromosome(chromo, inputs);

	showOutputs(chromo, num_outputs);

	freeDataSet(training_data);
	freeChromosome(chromo);
	freeParameters(params);

	chromosome *chromo_file = initialiseChromosomeFromFile("best_result.chromo");

	executeChromosome(chromo_file, inputs);

	showOutputs(chromo_file, num_outputs);

	delete [] inputs;
	delete [] outputs;
	return 0;
}


