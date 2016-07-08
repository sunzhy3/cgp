#ifndef NODEFUNCTION
#define NODEFUNCTION

/*基于马修斯相关系数(MCC)的fitness函数*/
double supervisedLearning_mcc(struct parameters *params, struct chromosome *chromo, struct dataSet *data);

/*向parameters中的functionSet添加函数*/
void add_functions(parameters* params);

/*初始化一个mat类型的指针，并为其中的mat附一个初值*/
Mat *initialise_mat(Mat *new_mat, int size);

void invert(Mat input, Mat& output);

void show_outputs(chromosome* chromo, int num_outputs);

/*文件处理和数据获取*/

//从文件中获取训练所需要的图片文件
void get_training_data(string file_path, Mat *inputs, Mat *outputs, int num_inputs, int num_outputs);

//从文件夹中读取文件名，并将找到的文件名都储存起来
void getFiles( string path, string exd, vector<string>& files);

//对视频文件中的帧进行抓取
int videocap(const string video);

/*自适应函数，用于SMCGP的函数，可以改变chromosome的构造的函数*/
Mat del_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights);
void _del_sm(chromosome* chromo, int node_index);

/*向to do list中写入smfunction的序号，作为执行时候寻找函数的依据*/
void set_todo(int *to_do, int func_num);

/*删除一个节点*/
void delete_node(chromosome *chromo, int node_index);

/*检查函数输入矩阵的type，并进行type转换，使函数可以正常进行操作*/
void check_convert(int num_inputs, Mat *inputs, int depth, int channels);

void motionToColor(Mat flow, Mat &color);
void makecolorwheel(vector<Scalar> &colorwheel);

/*计算函数，主要来自opencv3.1.0，对输入矩阵进行处理*/

/*自适应增加输入*/
Mat INP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);
Mat INPP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);
Mat SKIPINP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);

/*逻辑函数*/
Mat _and(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _or(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _nor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _xor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _xnor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _not(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);

/*空间处理*/
Mat _absdiff(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _min(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _min_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _max(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _max_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _avg(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _normalize(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);

/*阈值处理*/
Mat _threshold(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _thresholdInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _adaptiveThreshold(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _adaptiveThresholdInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*线性计算*/
Mat _subtract(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _subtract_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _add(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _add_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);

/*非线性计算*/
Mat _multiply(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _multiply_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _divide(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _divide_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _exp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _log(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _sqrt(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _pow(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*图像内容特征提取*/
Mat _goodFeaturesToTrack(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _sift(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _surf(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*图像内容边缘检测*/
Mat _Sobel(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Sobelx(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Sobely(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Canny(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Laplacian(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*线性与非线性滤波*/
Mat _blur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _boxFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _GaussianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _medianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _bilateralFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Gabor(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*形态学处理*/
Mat _erode(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _dilate(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _morphologyOpen(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyClose(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyTopHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyBlackHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyGradient(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*图像尺寸变换*/
Mat _pyrDown(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _pyrUp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*域变换*/
Mat _dctRows(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _dctInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*光学流*/
Mat _calcOpticalFlowFarneback(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights);
Mat _calcOpticalFlowSF(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights);
#endif