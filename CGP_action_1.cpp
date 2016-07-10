#include "stdafx.h"
#include "cgp_mat.h"
#include "node_function.h"

int main()
{
	struct parameters* params = NULL;
	struct dataSet* train_data = NULL;
	struct chromosome* chromo = NULL;

	int num_inputs = 3;
	int num_nodes = 50;
	int num_outputs = 1;
	int node_arity = 2;

	int num_gens = 100;
	double mutation_rate = 0.1;
	double target_fitness = 0.1;
	int update_frequency = 20;
	int num_threads = 4;

	params = initialiseParameters(num_inputs, num_nodes, num_outputs, node_arity);
	
	setCustomFitnessFunction(params, supervisedLearningIU, "supervisedLearningIU");

	setMutationRate(params, mutation_rate);
	
	setNumThreads(params, num_threads);

	setTargetFitness(params, target_fitness);

	setUpdateFrequency(params, update_frequency);

	printParameters(params);
	
	//allocate for training data 
	Mat *inputs = new Mat[(const int) num_inputs];
	Mat *outputs = new Mat[(const int) num_outputs];
	
	//vector<trainData> train_data;
	//getDataSetFromTrainData(traing_data, train_data[i]);

	string data_path = "/home/szy/cgp/image/VOCdevkit/VOC2012";

	//getTrainingData(file_path, inputs, outputs, num_inputs, num_outputs);

	training_data = initialiseDataSetFromFolder(data_path, num_inputs, num_outputs);

	chromo = runCGP(params, training_data, num_gens);

	printChromosome(chromo, 0);

	saveChromosome(chromo, "best_result.chromo");

	chromosome *chromo_file = initialiseChromosomeFromFile("best_result.chromo");

	executeChromosome(chromo_file, training_data->inputsData);

	showOutputs(chromo_file, num_outputs);
	
	freeDataSet(training_data);
	freeChromosome(chromo);
	freeParameters(params);

	delete [] inputs;
	delete [] outputs;
	return 0;
}


