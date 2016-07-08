#include "stdafx.h"

int main()
{
	struct parameters* params = NULL;
	struct dataSet* training_data = NULL;
	struct chromosome* chromo = NULL;

	int num_inputs = 2;
	int num_nodes = 50;
	int num_outputs = 2;
	int node_arity = 2;

	//allocate for training data 
	Mat *inputs = new Mat[(const int) num_inputs];
	Mat *outputs = new Mat[(const int) num_outputs];

	//capture frames from video file 
	//char* video = "F:/CGP_action_dataset/action_youtube_naudio/basketball/v_shooting_01";
	//int iscapped = videocap(video);

	//if(!iscapped){
	//	cout<<"Fail to capture frames from current video"<<endl;
	//	return 0;
	//}

	string file_path = "E:\CGP\image\v_shooting_01_01\training_data";

	get_training_data(file_path, inputs, outputs, num_inputs, num_outputs);

	int num_gens = 100;
	double target_fitness = 0.1;
	int update_frequency = 20;

	params = initialiseParameters(num_inputs, num_nodes, num_outputs, node_arity);

	setMutationRate(params, 0.1);

	setTargetFitness(params, target_fitness);

	setUpdateFrequency(params, update_frequency);

	printParameters(params);

	training_data = initialiseDataSetFromArrays(num_inputs, num_outputs, 1, inputs, outputs);

	chromo = runCGP(params, training_data, num_gens);

	printChromosome(chromo, 0);

	saveChromosome(chromo, "best_result.chromo");

	executeChromosome(chromo, inputs);

	show_outputs(chromo, num_outputs);

	freeDataSet(training_data);
	freeChromosome(chromo);
	freeParameters(params);

	chromosome *chromo_file = initialiseChromosomeFromFile("best_result.chromo");

	executeChromosome(chromo_file, inputs);

	show_outputs(chromo_file, num_outputs);

	delete [] inputs;
	delete [] outputs;
	return 0;
}


