#ifndef NODEFUNCTION
#define NODEFUNCTION

/*��������˹���ϵ��(MCC)��fitness����*/
double supervisedLearning_mcc(struct parameters *params, struct chromosome *chromo, struct dataSet *data);

/*��parameters�е�functionSet��Ӻ���*/
void add_functions(parameters* params);

/*��ʼ��һ��mat���͵�ָ�룬��Ϊ���е�mat��һ����ֵ*/
Mat *initialise_mat(Mat *new_mat, int size);

void invert(Mat input, Mat& output);

void show_outputs(chromosome* chromo, int num_outputs);

/*�ļ���������ݻ�ȡ*/

//���ļ��л�ȡѵ������Ҫ��ͼƬ�ļ�
void get_training_data(string file_path, Mat *inputs, Mat *outputs, int num_inputs, int num_outputs);

//���ļ����ж�ȡ�ļ����������ҵ����ļ�������������
void getFiles( string path, string exd, vector<string>& files);

//����Ƶ�ļ��е�֡����ץȡ
int videocap(const string video);

/*����Ӧ����������SMCGP�ĺ��������Ըı�chromosome�Ĺ���ĺ���*/
Mat del_sm(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *connectionWeights);
void _del_sm(chromosome* chromo, int node_index);

/*��to do list��д��smfunction����ţ���Ϊִ��ʱ��Ѱ�Һ���������*/
void set_todo(int *to_do, int func_num);

/*ɾ��һ���ڵ�*/
void delete_node(chromosome *chromo, int node_index);

/*��麯����������type��������typeת����ʹ���������������в���*/
void check_convert(int num_inputs, Mat *inputs, int depth, int channels);

void motionToColor(Mat flow, Mat &color);
void makecolorwheel(vector<Scalar> &colorwheel);

/*���㺯������Ҫ����opencv3.1.0�������������д���*/

/*����Ӧ��������*/
Mat INP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);
Mat INPP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);
Mat SKIPINP(const int numInputs, Mat *inputs, int parameters[], int *to_do, const double *weights);

/*�߼�����*/
Mat _and(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _or(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _nor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _xor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _xnor(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _not(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);

/*�ռ䴦��*/
Mat _absdiff(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _min(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _min_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _max(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _max_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _avg(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _normalize(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);

/*��ֵ����*/
Mat _threshold(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _thresholdInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _adaptiveThreshold(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _adaptiveThresholdInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*���Լ���*/
Mat _subtract(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _subtract_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _add(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);
Mat _add_c(const int numInputs, Mat *inputs, int parameters[],int *to_do, const double *weights);

/*�����Լ���*/
Mat _multiply(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _multiply_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _divide(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _divide_c(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _exp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _log(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _sqrt(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _pow(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*ͼ������������ȡ*/
Mat _goodFeaturesToTrack(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _sift(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _surf(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*ͼ�����ݱ�Ե���*/
Mat _Sobel(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Sobelx(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Sobely(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Canny(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Laplacian(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*������������˲�*/
Mat _blur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _boxFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _GaussianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _medianBlur(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _bilateralFilter(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _Gabor(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*��̬ѧ����*/
Mat _erode(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _dilate(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double *weights);
Mat _morphologyOpen(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyClose(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyTopHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyBlackHat(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _morphologyGradient(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*ͼ��ߴ�任*/
Mat _pyrDown(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _pyrUp(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*��任*/
Mat _dctRows(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);
Mat _dctInv(const int numInputs, Mat* inputs, int parameters[],int *to_do, const double* weights);

/*��ѧ��*/
Mat _calcOpticalFlowFarneback(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights);
Mat _calcOpticalFlowSF(const int numInputs, Mat* inputs, int parameters[], int *to_do, const double* weights);
#endif