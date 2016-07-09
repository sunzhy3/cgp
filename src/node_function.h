#ifndef NODEFUNCTION
#define NODEFUNCTION

double supervisedLearningMcc(struct parameters *params, struct chromosome *chromo, struct dataSet *data);

double supervisedLearningIU(struct parameters *params, struct chromosome *chromo, struct dataSet *data);

void addFunctions(parameters* params);

void invert(Mat input, Mat& output);

void showOutputs(chromosome* chromo, int num_outputs);

void getTrainingData(string file_path, Mat *inputs, Mat *outputs, int num_inputs, int num_outputs);

void getFiles( string path, string exd, vector<string>& files);

/*Mat del_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights);
void _del_sm(chromosome* chromo, int node_index);

void set_todo(int *to_do, int func_num);

void delete_node(chromosome *chromo, int node_index);*/

void check_convert(int num_inputs, Mat *inputs, int depth, int channels);

void motionToColor(Mat flow, Mat &color);

void makecolorwheel(vector<Scalar> &colorwheel);

Mat INP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);
Mat INPP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);
Mat SKIPINP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);

Mat _calcOpticalFlowFarneback(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights);
Mat _calcOpticalFlowSF(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights);

Mat _and(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _or(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _nor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _xor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _xnor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _not(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);

Mat _absdiff(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _min(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _min_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _max(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _max_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _avg(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _normalize(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);

Mat _threshold(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _thresholdInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _adaptiveThreshold(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _adaptiveThresholdInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _subtract(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _subtract_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _add(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _add_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);

Mat _multiply(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _multiply_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _divide(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _divide_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _exp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _log(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _sqrt(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _pow(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _goodFeaturesToTrack(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _sift(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _surf(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _Sobel(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Sobelx(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Sobely(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Canny(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Laplacian(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _blur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _boxFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _GaussianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _medianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _bilateralFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Gabor(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _erode(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _dilate(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _morphologyOpen(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyClose(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyTopHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyBlackHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyGradient(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _pyrDown(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _pyrUp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _dctRows(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _dctInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

Mat _transpose(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _flipx(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _flipy(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
#endif
