#include"stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <iomanip>
#include <math.h>
#include <xfeatures2d/nonfree.hpp>
#include <optflow/motempl.hpp>
#include <iostream>
#include <vector>
#include <string.h>

using namespace cv;
using namespace std;

#define FLOW_THRESH 10

Mat *initialise_mat(Mat *new_mat, int size) {

	const int mat_size = size;
	new_mat = new Mat[mat_size];

	for(int i = 0; i < mat_size; i++){
		new_mat[i] = Mat::zeros(3, 3, CV_8UC1);
	}

	return new_mat;
}

void show_outputs(chromosome* chromo, int num_outputs) {

	for(int i = 0; i < num_outputs; i++) {
		char name[21] = "best result ";
		char number[3];
		_itoa_s(i, number, 10);
		strcat_s(name, number);
		namedWindow(name);
		imshow(name, chromo->outputValues[i]);
	}
	waitKey(0);
}

/*get training data from folder*/
void get_training_data(string file_path, Mat *inputs, Mat *outputs, int num_inputs, int num_outputs) {
	
	vector<string> files;

	getFiles(file_path, "jpg", files);

	for(int i = 0; i < num_inputs + num_outputs; i++){

		if(i <num_inputs){
			inputs[i] = imread(files[i], 0);
		}
		else{
			outputs[i - num_inputs] = imread(files[i], 0);
		}
	}
}

/*get file's name from path, the file must have the special extend, the name will be restored into the vector files*/
void getFiles( string path, string exd, vector<string>& files) {

	long hFile = 0;
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), "") ) {
		exdName = "\\*." + exd;
	}
	else{
		exdName = "\\*";
	}

	if((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(),&fileinfo) ) !=  -1) {

		do{
			//if there is subfolder, iterate it 
			if((fileinfo.attrib &  _A_SUBDIR) ) {
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0) {
					getFiles( pathName.assign(path).append("\\").append(fileinfo.name), exd, files );
				}
			}
			else {
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0) {
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
				}
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}


/*capture frames from video and save them for jpg*/
int videocap(const string video) {

	Mat frame;
	Mat gray_image;
	string video_file = video;
	string image;//path of the image file
	string extend = ".jpg";
	struct _finddata_t fileinfo;
	
	_findfirst(video_file.c_str(), &fileinfo);
	
	VideoCapture cap(0);
	cap.open(video);

	if (!cap.isOpened()){
		return 0; 
	}

	double frame_number = cap.get(CV_CAP_PROP_FRAME_COUNT);

	for(int i = 1; i < frame_number; i++) {

		cap.read(frame);

		//convert RGB image to gray image
		gray_image = frame.clone();
		cvtColor(gray_image, gray_image, CV_BGR2GRAY, 1);

		//save the image
		string folder_path, save_path, folder_name;
		folder_name.assign(fileinfo.name);
		save_path = "E\\study\\CGP_project\\image\\" + folder_name;
		folder_path = "md " + save_path;
		system(folder_path.c_str());//create a new folder
		
		char frame_count[8];
		_itoa_s(i, frame_count, 255, 10);
		
		image = save_path.append("\\").append(folder_name).append(frame_count).append(extend);
		
		imwrite(image, gray_image);//save the gray image

		//test by read the image file and show it on the screen
		Mat current_frame = imread(image);
		imshow("current_frame", current_frame);
		waitKey(10000);

		/*capture next frame*/
		cap.set(CV_CAP_PROP_POS_FRAMES, (double)i );
	}
	return 1;
}

void add_functions(parameters* params) {

	add_smfunction(params, _del_sm);

	addCustomNodeFunction(params, INP, "INP", 8);
	addCustomNodeFunction(params, INPP, "INPP", 8);
	addCustomNodeFunction(params, SKIPINP, "SKIPINP", 8);
	addCustomNodeFunction(params, _calcOpticalFlowFarneback, "optflfraneback", 2);
	addCustomNodeFunction(params, _calcOpticalFlowSF, "optflsf", 2);

	addCustomNodeFunction(params, _and, "and", 2);
	addCustomNodeFunction(params, _or, "or", 2);
	addCustomNodeFunction(params, _nor, "nor", 2);
	addCustomNodeFunction(params, _xor, "xor", 2);
	addCustomNodeFunction(params, _xnor, "xnor", 2);
	addCustomNodeFunction(params, _not, "not", 2);
	addCustomNodeFunction(params, _min, "min", 2);
	addCustomNodeFunction(params, _min_c, "minc", 1);
	addCustomNodeFunction(params, _max, "max", 2);
	addCustomNodeFunction(params, _max_c, "maxc", 1);
	addCustomNodeFunction(params, _absdiff, "absdiff", 2);
	addCustomNodeFunction(params, _normalize, "normalize", 1);

	addCustomNodeFunction(params, _avg, "avg", 2);
	addCustomNodeFunction(params, _add, "add", 2);
	addCustomNodeFunction(params, _add_c, "addc", 1);
	addCustomNodeFunction(params, _subtract, "sub", 2);
	addCustomNodeFunction(params, _subtract_c, "subc", 1);
	addCustomNodeFunction(params, _multiply, "mul", 2);
	addCustomNodeFunction(params, _multiply_c, "mulc", 1);
	addCustomNodeFunction(params, _divide, "div", 2);
	addCustomNodeFunction(params, _divide_c, "divc", 1);
	addCustomNodeFunction(params, _exp, "exp", 1);
	addCustomNodeFunction(params, _log, "log", 1);
	addCustomNodeFunction(params, _pow, "pow", 1);
	addCustomNodeFunction(params, _sqrt, "sqrt", 1);

	addCustomNodeFunction(params, _goodFeaturesToTrack, "goodFTT", 1);
	addCustomNodeFunction(params, _sift, "sift", 1);
	addCustomNodeFunction(params, _surf, "surf", 1);

	addCustomNodeFunction(params, _Sobel, "Sobel", 1);
	addCustomNodeFunction(params, _Sobelx, "Sobelx", 1);
	addCustomNodeFunction(params, _Sobely, "Sobely", 1);
	addCustomNodeFunction(params, _Canny, "Canny", 1);
	addCustomNodeFunction(params, _Laplacian, "Laplacian", 1);

	addCustomNodeFunction(params, _blur, "blur", 1);
	addCustomNodeFunction(params, _boxFilter, "boxFilter", 1);
	addCustomNodeFunction(params, _GaussianBlur, "GaussianBlur", 1);
	addCustomNodeFunction(params, _medianBlur, "medianBlur", 1);
	addCustomNodeFunction(params, _bilateralFilter, "bilateral", 1);

	addCustomNodeFunction(params, _erode, "erode", 1);
	addCustomNodeFunction(params, _dilate, "dilate", 1);
	addCustomNodeFunction(params, _morphologyOpen, "morOpen", 1);
	addCustomNodeFunction(params, _morphologyClose, "morClose", 1);
	addCustomNodeFunction(params, _morphologyTopHat, "morTopHat", 1);
	addCustomNodeFunction(params, _morphologyBlackHat, "morBlackhat", 1);
	addCustomNodeFunction(params, _morphologyGradient, "morGradient", 1);

	addCustomNodeFunction(params, _pyrDown, "pyrDown", 1);
	addCustomNodeFunction(params, _pyrUp, "pyrUp", 1);

	addCustomNodeFunction(params, _threshold, "thresh", 1);
	addCustomNodeFunction(params, _thresholdInv, "threshInv", 1);
	addCustomNodeFunction(params, _adaptiveThreshold, "adThresh", 1);
	addCustomNodeFunction(params, _adaptiveThresholdInv, "adThreshInv", 1);

	addCustomNodeFunction(params, _Gabor, "Gabor", 1);
	addCustomNodeFunction(params, _dctRows, "dctRows", 1);
	addCustomNodeFunction(params, _dctInv, "dctInv", 1);

	addCustomNodeFunction(params, del_sm, "del_sm", 1);
}

double supervisedLearning_mcc(struct parameters *params, struct chromosome *chromo, struct dataSet *data) {

	double fitness = 0;

	/* error checking */
	if (getNumChromosomeInputs(chromo) != getNumDataSetInputs(data)) {
		printf("Error: the number of chromosome inputs must match the number of inputs specified in the dataSet.\n");
		printf("Terminating CGP-Library.\n");
		exit(0);
	}

	if (getNumChromosomeOutputs(chromo) != getNumDataSetOutputs(data)) {
		printf("Error: the number of chromosome outputs must match the number of outputs specified in the dataSet.\n");
		printf("Terminating CGP-Library.\n");
		exit(0);
	}

	/* for each sample in data */
	for (int i = 0 ; i < getNumDataSetSamples(data); i++) {

		double tp = 1, tn = 1, fp = 1, fn = 1;
		double mcc = 0;
		/* calculate the chromosome outputs for the set of inputs  */
		executeChromosome(chromo, getDataSetSampleInputs(data, i));

		/* for each chromosome output */
		for (int j = 0; j < getNumChromosomeOutputs(chromo); j++) {

			Mat output_mat = getChromosomeOutput(chromo, j).clone();
			Mat target_mat = getDataSetSampleOutput(data, i, j).clone();
			Mat test[2] = {target_mat, output_mat};

	        check_convert(2, test, CV_8U, 1);

			MatIterator_<uchar> out = test[0].begin<uchar>();
			MatIterator_<uchar> tar = test[1].begin<uchar>();

			for(; tar != test[1].end<uchar>(); out++, tar++) {

					if(*out != 0 && *tar != 0) {
				        tp = tp + 1;
			        } else if(*out == 0 && *tar != 0) {
				        fn = fn +1;
			        } else if(*out != 0 && *tar == 0) {
						fp = fp + 1;
					} else {
						tn = tn + 1;
					}
			}
		}
		mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp +fn) * (tn + fp) * (tn +fn));
		fitness += abs(1 - abs(mcc));
	}
	fitness = fitness / getNumDataSetSamples(data);

	return fitness;
}

//检测输入的矩阵的类型和大小是否匹配,如果不满足要求的话就进行转换
void check_convert(int num_inputs, Mat *inputs, int depth, int channels)
{
	if(num_inputs == 2) {
		if(inputs[0].rows != inputs[1].rows || inputs[0].cols != inputs[1].cols) {
			resize(inputs[1], inputs[1], Size(inputs[0].cols, inputs[0].rows));
		}
		if(inputs[0].depth() != depth) {
			inputs[0].convertTo(inputs[0], depth);
		}
		if(inputs[1].depth() != depth) {
			inputs[1].convertTo(inputs[1], depth);
		}
		if(inputs[0].channels() != channels) {
			cvtColor(inputs[0], inputs[0], CV_BGR2GRAY, channels);
		}
		if(inputs[1].channels() != 1) {
			cvtColor(inputs[1], inputs[1], CV_BGR2GRAY, channels);
		}
	} else {
		if(inputs[0].depth() != depth) {
			inputs[0].convertTo(inputs[0], depth);
		}
		if(inputs[0].channels() != channels) {
			cvtColor(inputs[0], inputs[0], CV_BGR2GRAY, channels);
		}
	}
}

Mat INP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *connectionWeights) {

	int current_input = to_do[5];
	Mat result = inputs[current_input].clone();

	current_input = current_input + 1;

	if(current_input >= numInputs) {
		current_input = 0;
	}
	to_do[5] = current_input;
	return result;
}

Mat INPP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *connectionWeights) {

	int current_input = to_do[5];
	Mat result = inputs[current_input].clone();

	current_input = current_input - 1;

	if(current_input < 0) {
		current_input = 0;
	}
	to_do[5] = current_input;
	return result;
}

Mat SKIPINP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *connectionWeights) {

	int current_input = to_do[5];
	Mat result = inputs[current_input].clone();

	current_input = current_input + parameters[1];
	current_input = current_input % numInputs;
	to_do[5] = current_input;

	return result;
}

void set_todo(int *to_do, int func_num) {

	int num = to_do[2];
	to_do[num] = func_num;
}

void delete_node(chromosome *chromo, int node_index) {

	int numInputs = chromo->numInputs;

	for(int j = node_index - numInputs; j < chromo->numNodes - 1; j++){
		copyNode(chromo->nodes[j + 1], chromo->nodes[j]);
	}

	freeNode(chromo->nodes[chromo->numNodes - 1]);
	chromo->numNodes--;

	for(int i = 0; i < chromo->numOutputs; i++) {

		if(chromo->outputNodes[i] >= chromo->numNodes + numInputs) {
			chromo->outputNodes[i]--;
		}
	}
}

void _del_sm(chromosome* chromo, int node_index) {

	int node_location = node_index - chromo->numInputs;
	int p1 = chromo->nodes[node_location]->func_params[1];
	int p2 = chromo->nodes[node_location]->func_params[2];
	int low_bound = node_index + p1;
	int up_bound = node_index + p1 + p2;

	if(up_bound > chromo->numNodes){
		up_bound = chromo->numNodes;
	}

	for(int i = low_bound; i <= up_bound; i++){

		delete_node(chromo, i);
	}
}

Mat del_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	set_todo(to_do, 0);

	return inputs[0];
}

Mat add_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	set_todo(to_do, 1);

	return inputs[0];
}

Mat mov_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	set_todo(to_do, 2);

	return inputs[0];
}

Mat ovr_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	set_todo(to_do, 3);

	return inputs[0];
}

Mat dup_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	set_todo(to_do, 4);

	return inputs[0];
}

Mat dup2_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	set_todo(to_do, 5);

	return inputs[0];
}

Mat dup3_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	set_todo(to_do, 6);

	return inputs[0];
}

/*计算函数*/

//取两个矩阵差值的绝对值存在输出矩阵中
Mat _absdiff(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;

	check_convert(numInputs, inputs, CV_8U, 1);

	absdiff(inputs[0], inputs[1], output);

	return output;
}

//形态学处理，对矩阵进行腐蚀
Mat _erode(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	int iterations = parameters[1];

	erode(inputs[0], output, Mat(), Point(-1, -1), iterations);

	return output;
}

//形态学处理，对矩阵进行膨胀处理
Mat _dilate(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	int iterations = parameters[1];

	dilate(inputs[0], output, Mat(), Point(-1, -1), iterations);

	return output;
}

//比较两个矩阵的对应元素，并保留小的那个
Mat _min(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	
	check_convert(numInputs, inputs, CV_8U, 1);

	min(inputs[0], inputs[1], output);

	return output;
}

//比较两个矩阵的对应元素与一个常数,保留两者中更小的
Mat _min_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	double min_const = parameters[0];
	
	check_convert(numInputs, inputs, CV_8U, 1);

	min(inputs[0], Scalar(min_const), output);

	return output;
}

//比较两个矩阵的对应元素，并保留大的那个
Mat _max(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;

	check_convert(numInputs, inputs, CV_8U, 1);

	max(inputs[0], inputs[1], output);

	return output;
}

//比较两个矩阵的对应元素与一个常数,保留两者中更大的
Mat _max_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	double max_const = (double) parameters[0];

	check_convert(numInputs, inputs, CV_8U, 1);

	max(inputs[0], Scalar(max_const), output);

	return output;
}

//and函数，当两个输入矩阵对应元素中有0值时，output对应元素为0，此外output对应元素为输入矩阵元素之和
Mat _and(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	check_convert(numInputs, inputs, CV_8U, 1);
	Mat output = inputs[0].clone();

	MatIterator_<uchar> inputs_0 = inputs[0].begin<uchar>();
	MatIterator_<uchar> inputs_1 = inputs[1].begin<uchar>();
	MatIterator_<uchar> out = output.begin<uchar>();

	for(;inputs_0 != inputs[0].end<uchar>(); inputs_0++, inputs_1++, out++) {
		if(*inputs_0 != 0 && *inputs_1 != 0) {
			*out = *inputs_0 + *inputs_1;
		}
		else {
			*out = 0;
		}
	}
	return output;
}

//or函数，当两个输入矩阵对应元素全为0值时，output对应元素为0，此外output对应元素为输入矩阵元素之和
Mat _or(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	check_convert(numInputs, inputs, CV_8U, 1);
	Mat output = inputs[0].clone();

	MatIterator_<uchar> inputs_0 = inputs[0].begin<uchar>();
	MatIterator_<uchar> inputs_1 = inputs[1].begin<uchar>();
	MatIterator_<uchar> out = output.begin<uchar>();

	for(inputs_0 = inputs[0].begin<uchar>(), out = output.begin<uchar>(), inputs_1 = inputs[1].begin<uchar>();
		inputs_0 != inputs[0].end<uchar>(); inputs_0++, inputs_1++, out++) {

		if(*inputs_0 != 0 && *inputs_1 != 0) {
			*out = 0;
		}
		else {
			*out = *inputs_0 + *inputs_1;
		}
	}
	return output;
}

//nor函数，当两个输入矩阵对应元素不全为0值时，output对应元素为0，此外output对应元素为输入矩阵元素之和
Mat _nor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	check_convert(numInputs, inputs, CV_8U, 1);
	Mat output = inputs[0].clone();

	MatIterator_<uchar> inputs_0 = inputs[0].begin<uchar>();
	MatIterator_<uchar> inputs_1 = inputs[1].begin<uchar>();
	MatIterator_<uchar> out = output.begin<uchar>();

	for(inputs_0 = inputs[0].begin<uchar>(), out = output.begin<uchar>(), inputs_1 = inputs[1].begin<uchar>();
		inputs_0 != inputs[0].end<uchar>(); inputs_0++, inputs_1++, out++) {
		if(*inputs_0 != 0 || *inputs_1 != 0) {
			*out = 0;
		}
		else {
			*out = *inputs_0 + *inputs_1;
		}
	}
	return output;
}

//xor函数，当两个输入矩阵对应元素全为0值或全不为0时，output对应元素为输入矩阵元素之和，此外output对应元素为0
Mat _xor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	check_convert(numInputs, inputs, CV_8U, 1);
	Mat output = inputs[0].clone();

	int test = inputs[0].elemSize();

	MatIterator_<uchar> inputs_0 = inputs[0].begin<uchar>();
	MatIterator_<uchar> inputs_1 = inputs[1].begin<uchar>();
	MatIterator_<uchar> out = output.begin<uchar>();

	for(/*inputs_0 = inputs[0].begin<uchar>(), inputs_1 = inputs[1].begin<uchar>(), out = output.begin<uchar>()*/;
		inputs_0 != inputs[0].end<uchar>(); inputs_0++, inputs_1++, out++) {

		if((*inputs_0 != 0 && *inputs_1 != 0) || (*inputs_0 == 0 && *inputs_1 == 0)) {
			*out = 0;
		} else {
			*out = *inputs_0 + *inputs_1;
		}
	}
	return output;
}

//xnor函数，当两个输入矩阵对应元素全不为0或全为0时，output对应元素为输入矩阵元素之和，此外output对应元素为0
Mat _xnor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	check_convert(numInputs, inputs, CV_8U, 1);
	Mat output = inputs[0].clone();

	MatIterator_<uchar> inputs_0 = inputs[0].begin<uchar>();
	MatIterator_<uchar> inputs_1 = inputs[1].begin<uchar>();
	MatIterator_<uchar> out = output.begin<uchar>();

	for(/*inputs_0 = inputs[0].begin<uchar>(), out = output.begin<uchar>(), inputs_1 = inputs[1].begin<uchar>()*/;
		inputs_0 != inputs[0].end<uchar>(); inputs_0++, inputs_1++, out++) {

		if((*inputs_0 != 0 && *inputs_1 != 0) || (*inputs_0 == 0 && *inputs_1 == 0)) {
			*out = *inputs_0 + *inputs_1;
		} else {
			*out = 0;
		}
	}
	return output;
}

//not函数，对输入矩阵求反
Mat _not(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	check_convert(numInputs, inputs, CV_8U, 1);
	Mat output = inputs[0].clone();

	MatIterator_<uchar> inputs_0 = inputs[0].begin<uchar>();
	MatIterator_<uchar> inputs_1 = inputs[1].begin<uchar>();
	MatIterator_<uchar> out = output.begin<uchar>();

	for(;inputs_0 != inputs[0].end<uchar>(); inputs_0++, out++) {

		*out = abs(255 - *inputs_0);
	}
	return output;
}

//对矩阵进行归一化
Mat _normalize(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;

	normalize( inputs[0], output );

	return output;
}

//对矩阵进行二值化，阈值由一个输入矩阵元素的平均值决定
Mat _threshold(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	double thresh = (double) parameters[0];

	threshold( inputs[0], output, thresh, 1, THRESH_BINARY );

	return output;
}

//对矩阵进行二值化，阈值由一个输入矩阵元素的平均值决定
Mat _thresholdInv(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	double thresh = (double) parameters[0];

	threshold(inputs[0], output, thresh, 1, THRESH_BINARY_INV);

	return output;
}

//两个矩阵做平均
Mat _avg(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;

	check_convert(numInputs, inputs, CV_8U, 1);
	
	addWeighted(inputs[0], 0.5, inputs[1],0.5, 0, output);

	min(output, Scalar(255), output);

	return output;
}

//两个矩阵相加
Mat _add(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;

	check_convert(numInputs, inputs, CV_8U, 1);
	
	add(inputs[0], inputs[1], output);

	min(output, Scalar(255), output);

	return output;
}

//矩阵加上一个常数
Mat _add_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	double add_const = (double) parameters[0];

	check_convert(numInputs, inputs, CV_8U, 1);
	
	add(inputs[0], Scalar(add_const), output);

	min(output, Scalar(255), output);

	return output;
}

//两个矩阵相减
Mat _subtract(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;

	check_convert(numInputs, inputs, CV_8U, 1);
	
	subtract(inputs[0], inputs[1], output);

	max(output, Scalar(0), output);

	return output;
}

//矩阵减去一个常数
Mat _subtract_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;
	double sub_const = (double) parameters[0];

	check_convert(numInputs, inputs, CV_8U, 1);
	
	subtract(inputs[0], Scalar(sub_const), output);

	max(output, Scalar(0), output);

	return output;
}

//将两个矩阵相乘，要求进行运算的两个矩阵有相同的大小和类型
Mat _multiply(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights) {

	Mat output;

	check_convert(numInputs, inputs, CV_8U, 1);

	multiply(inputs[0], inputs[1], output);

	min(output, Scalar(255), output);

	return output;

}

//将矩阵中的每个元素都乘以一个常数
Mat _multiply_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights) {

	Mat output;
	double mul_const = (double) parameters[1];

	check_convert(numInputs, inputs, CV_8U, 1);

	multiply(inputs[0], Scalar(mul_const), output);

	min(output, Scalar(255), output);

	return output;

}

//将两个矩阵相除,要求进行运算的两个矩阵有相同的大小和类型
Mat _divide(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	check_convert(numInputs, inputs, CV_8U, 1);

	divide(inputs[0],inputs[1],output);

	return output;
}

//将矩阵中的元素除以一个常数
Mat _divide_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	double div_const = (double) parameters[1] + 1;

	check_convert(numInputs, inputs, CV_8U, 1);

	divide(inputs[0],Scalar(div_const),output);

	return output;
}

//计算矩阵每个元素的e指数阶
Mat _exp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	check_convert(numInputs, inputs, CV_32F, 1);

	exp(inputs[0],output);

	output.convertTo(output, CV_8U);

	return output;
}

//计算矩阵每个元素的自然对数
Mat _log(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	check_convert(numInputs, inputs, CV_32F, 1);

	log(inputs[0],output);

	output.convertTo(output, CV_8U);

	return output;
}

//计算矩阵每个元素的平方
Mat _pow(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	double power = 2.0;

	check_convert(numInputs, inputs, CV_32F, 1);

	pow(inputs[0], power, output);

	output.convertTo(output, CV_8U);

	return output;
}

//计算矩阵每个元素的平方根
Mat _sqrt(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	check_convert(numInputs, inputs, CV_32F, 1);

	sqrt(inputs[0], output);

	output.convertTo(output, CV_8U);

	return output;
}

//定位变化剧烈的特征点，并以向量形式返回
//mask为空，blocksize为默认值3，useHarrisDetector为false，参数k为默认值0.04
Mat _goodFeaturesToTrack(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	vector<Point2f> corners;
	int maxCorners = parameters[0];
	double qualityLevel = 0.01;
	double minDistance = (double) parameters[1];

	check_convert(numInputs, inputs, CV_8U, 1);
	
	goodFeaturesToTrack(inputs[0], corners, maxCorners, qualityLevel, minDistance);

	output = inputs[0].clone();

	for(unsigned int i = 0; i < corners.size(); i++){
		circle(output, corners[i], 4, Scalar(255));
	}

	return output;
}

//sobel边缘提取算子,将x与y方向的梯度权值相加
Mat _Sobel(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output, grad_x, grad_y;

	Sobel(inputs[0], grad_x, inputs[0].depth(), 1, 0);

	Sobel(inputs[0], grad_y, inputs[0].depth(), 0, 1);

	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, output);

	return output;
}

//sobel边缘提取算子，dx和dy分别为1和0，sobel核的大小ksize为默认3*3
Mat _Sobelx(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int ksize = parameters[1];

	if(parameters[1] % 2 == 0){
		ksize = parameters[1] + 1;
	}

	Sobel(inputs[0], output, inputs[0].depth(), 1, 0, ksize);

	return output;
}

//sobel边缘提取算子，dx和dy分别为0和1，sobel核的大小ksize为默认3*3
Mat _Sobely(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int ksize = parameters[1];

	if(parameters[1] % 2 == 0){
		ksize = parameters[1] + 1;
	}

	Sobel(inputs[0], output, inputs[0].depth(), 0, 1, ksize);

	return output;
}

//均值滤波，kernel大小为3*3
Mat _blur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	blur(inputs[0], output, Size(parameters[1], parameters[2]));

	return output;
}

//方框滤波，kernel大小为3*3
Mat _boxFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	boxFilter(inputs[0], output, inputs[0].depth(), Size(parameters[1], parameters[2]));

	return output;
}

//高斯滤波，kernel大小为3*3，x和y方向上的标准差根据kernel的大小来计算
Mat _GaussianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int ksize_width = parameters[1];
	int ksize_height = parameters[2];

	if(parameters[1] % 2 == 0){
		ksize_width = parameters[1] + 1;
	}
	if(parameters[2] % 2 == 0){
		ksize_height = parameters[2] + 1;
	}

	GaussianBlur(inputs[0], output, Size(ksize_width,ksize_height), 0, 0);

	return output;
}

//中值滤波，kernel大小为3*3
Mat _medianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int ksize = parameters[1];

	medianBlur(inputs[0], output, 3);

	return output;
}

//双边滤波，设置参数d为3，sigmaColor为1，sigmaSpace为1
Mat _bilateralFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	bilateralFilter(inputs[0], output, 3, 1, 1);

	return output;
}

//sift特征提取,将提取出来的特征点标为白色
Mat _sift(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	vector<KeyPoint> keyPoint;
	Ptr<Feature2D> detector = xfeatures2d::SIFT::create(500);

	detector->detect(inputs[0], keyPoint);

	detector->compute(inputs[0], keyPoint, output);

	drawKeypoints(inputs[0], keyPoint, output, Scalar(255));

	output.convertTo(output, CV_8UC1);

	return output;
}

//surf特征提取,将提取出来的特征点标为白色
Mat _surf(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	vector<KeyPoint> keyPoint;
	Ptr<Feature2D> detector = xfeatures2d::SURF::create(500);

	detector->detect(inputs[0], keyPoint);

	detector->compute(inputs[0], keyPoint, output);

	drawKeypoints(inputs[0], keyPoint, output, Scalar(255));

	output.convertTo(output, CV_8UC1);

	return output;
}

//形态学处理open，先腐蚀后膨胀
Mat _morphologyOpen(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	Mat kernel = getStructuringElement(MORPH_RECT, Size(parameters[1], parameters[2]));

	morphologyEx(inputs[0], output, MORPH_OPEN, kernel);

	return output;
}

//形态学处理close，先膨胀后腐蚀
Mat _morphologyClose(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	Mat kernel = getStructuringElement(MORPH_RECT, Size(parameters[1], parameters[2]));

	morphologyEx(inputs[0], output, MORPH_CLOSE, kernel);

	return output;
}

//形态学处理，顶帽处理为原图像减去open处理的结果
Mat _morphologyTopHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	Mat kernel = getStructuringElement(MORPH_RECT, Size(parameters[1], parameters[2]));

	morphologyEx( inputs[0], output, MORPH_TOPHAT, kernel );

	return output;
}

//形态学处理，黒帽处理为close处理的结果减去原图像
Mat _morphologyBlackHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(parameters[1], parameters[2]));

	morphologyEx(inputs[0], output, MORPH_BLACKHAT, kernel);

	return output;
}

//形态学处理，梯度处理为膨胀结果减去腐蚀结果
Mat _morphologyGradient(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(parameters[1], parameters[2]));

	morphologyEx(inputs[0], output, MORPH_GRADIENT, kernel);

	return output;
}

//产生高斯金字塔（由高斯函数进行滤波之后再进行下采样），并进行图像缩小，变为原来的一半
Mat _pyrDown(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int cols = inputs[0].cols;
	int rows = inputs[0].rows;

	pyrDown(inputs[0], output);

	resize(output, output, Size(cols, rows) );

	return output;
}

//产生高斯金字塔（由高斯函数进行滤波之后再进行下采样），并进行图像扩张，变为原来的2倍
Mat _pyrUp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int cols = inputs[0].cols;
	int rows = inputs[0].rows;

	pyrUp(inputs[0], output);

	resize(output, output, Size(cols, rows) );

	return output;
}

//自适应阈值二值化处理，最大值为255，阈值选择方式为(x,y) 的附近 blockSize * blockSize 的（使用高斯分布）加重平均的值减去C的值．
//从blockSize确定这高斯分布的标准偏差，二值化方式为正向，blockSize为奇数且大于1，参数C设为0
Mat _adaptiveThreshold(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int block_size = parameters[1];

	if(parameters[1] % 2 == 0){
		block_size = parameters[1] + 1;
	}
	else{
		block_size = parameters[1] + 2;
	}

	check_convert(numInputs, inputs, CV_8U, 1);

	adaptiveThreshold(inputs[0], output, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block_size, 0);

	return output;
}

/*反向自适应阈值二值化处理，最大值为255，阈值选择方式为(x,y) 的附近 blockSize * blockSize 的（使用高斯分布）加重平均的值减去C的值．
从blockSize确定这高斯分布的标准偏差，二值化方式为反向，blockSize为3，参数C设为0*/
Mat _adaptiveThresholdInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	int block_size = parameters[1];

	if(parameters[1] % 2 == 0){
		block_size = parameters[1] + 1;
	}
	else{
		block_size = parameters[1] + 2;
	}

	check_convert(numInputs, inputs, CV_8U, 1);

	adaptiveThreshold(inputs[0], output, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block_size, 0);

	return output;
}

//canny边缘检测，低阈值选择为10，高阈值选择为30，sobel算子选择默认的3 * 3大小，L2gradient默认为false
Mat _Canny(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	double threshold = parameters[1];

	Canny(inputs[0], output, threshold, threshold * 3);

	return output;
}

//拉普拉斯变换
Mat _Laplacian(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	Laplacian(inputs[0], output, inputs[0].depth(), 3);

	return output;
}

//Gabor变换
Mat _Gabor(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;
	double sigma = 1;
	double theta = (double)parameters[4];
	double lambd = (double)parameters[3];
	double gamma = (double)randDecimal();

	Mat gaborKernel = getGaborKernel(Size(parameters[1], parameters[2]), sigma, theta, lambd, gamma);

	filter2D(inputs[0], output, inputs[0].depth(), gaborKernel);

	return output;
}

//DCT反变换
Mat _dctInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	check_convert(numInputs, inputs, CV_32F, 1);

	dct(inputs[0], output, DCT_INVERSE);

	output.convertTo(output, CV_8U);

	return output;
}

//DCT变换
Mat _dctRows(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights) {

	Mat output;

	check_convert(numInputs, inputs, CV_32F, 1);

	dct(inputs[0], output, DCT_ROWS);

	output.convertTo(output, CV_8U);

	return output;
}

/*稠密光学流*/
Mat _calcOpticalFlowFarneback(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights) {

	check_convert(numInputs, inputs, CV_8U, 1);

	Mat flow(inputs[0].rows, inputs[0].cols, CV_32FC2);
	Mat output, result;
	double pyr_scale = 0.5;
	int levels = 3;
	int winsize = 5;
	int iterations = 3;
	int poly_n = 5;
	double poly_sigma = 1.1;

	calcOpticalFlowFarneback(inputs[0], inputs[1], flow, pyr_scale, levels, winsize,iterations, poly_n, poly_sigma, OPTFLOW_FARNEBACK_GAUSSIAN);

	/*将动作向量转换成图像中的色彩强度*/
	motionToColor(flow, result);

	result.convertTo(result, CV_8U);
	cvtColor(result, result, CV_BGR2GRAY, 1);
	invert(result, output);

	return output;
}

/*稠密光学流基础上的简单光学流*/
Mat _calcOpticalFlowSF(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights) {

	check_convert(numInputs, inputs, CV_8U, 3);

	Mat flow(inputs[0].rows, inputs[0].cols, CV_32FC2);
	Mat output, result;
	int layers = 3;
	int averaging_block_size = 2;
	int max_flow = 4;

	optflow::calcOpticalFlowSF(inputs[0], inputs[1], flow, layers, averaging_block_size, max_flow);

	//将动作向量转换成图像中的色彩强度
	motionToColor(flow, result);

	result.convertTo(result, CV_8U);
	cvtColor(result, result, CV_BGR2GRAY, 1);
	invert(result, output);

	return output;
}

/*色轮采用MunsellColor System*/
void makecolorwheel(vector<Scalar> &colorwheel) {
	int RY = 15;  
	int YG = 6;  
	int GC = 4;  
	int CB = 11;  
	int BM = 13;  
	int MR = 6;  

	int i;  

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));  
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));  
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));  
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));  
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));  
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));  
}  

/*将光流向量转换成颜色强度图像*/
void motionToColor(Mat flow, Mat &color) {
	if (color.empty())  
		color.create(flow.rows, flow.cols, CV_8UC3);  

	static vector<Scalar> colorwheel; //标量R,G,B
	if (colorwheel.empty())  
		makecolorwheel(colorwheel);  

	//决定动作范围
	float maxrad = -1;  

	//寻找最大的流来对fx和fy归一化
	for (int i= 0; i < flow.rows; ++i)   
	{  
		for (int j = 0; j < flow.cols; ++j)   
		{  
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
			float fx = flow_at_point[0];  
			float fy = flow_at_point[1];  
			if ((fabs(fx) >  FLOW_THRESH) || (fabs(fy) > FLOW_THRESH))  
				continue;  
			float rad = sqrt(fx * fx + fy * fy);  
			maxrad = maxrad > rad ? maxrad : rad;  
		}  
	}  

	for (int i= 0; i < flow.rows; ++i)   
	{  
		for (int j = 0; j < flow.cols; ++j)   
		{  
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);  

			float fx = flow_at_point[0] / maxrad;  
			float fy = flow_at_point[1] / maxrad;  
			if ((fabs(fx) >  FLOW_THRESH) || (fabs(fy) > FLOW_THRESH))  
			{  
				data[0] = data[1] = data[2] = 0;  
				continue;  
			}  
			float rad = sqrt(fx * fx + fy * fy);  

			double angle = atan2(-fy, -fx) / CV_PI;  
			double fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);
			int k0 = (int)fk;  
			int k1 = (k0 + 1) % colorwheel.size();  
			double f = fk - k0;  
			//f = 0; // uncomment to see original color wheel

			for (int b = 0; b < 3; b++)   
			{  
				double col0 = colorwheel[k0][b] / 255.0;  
				double col1 = colorwheel[k1][b] / 255.0;  
				double col = (1 - f) * col0 + f * col1;  
				if (rad <= 1)  
					col = 1 - rad * (1 - col); // increase saturation with radius
				else  
					col *= .75; //超出范围
				data[2 - b] = (int)(255.0 * col);  
			}  
		}  
	}  
}

/*将图像的颜色翻转*/
void invert(Mat input, Mat& output) {

	if(input.depth() != CV_8U) {
		input.convertTo(input, CV_8U);
	}
	if(input.channels() != 1) {
		cvtColor(input, input, CV_BGR2GRAY, 1);
	}
	output = input.clone();

	MatIterator_<uchar> in = input.begin<uchar>();
	MatIterator_<uchar> out = output.begin<uchar>();

	for(;in != input.end<uchar>(); in++, out++) {

		*out = abs(255 - *in);
	}
}