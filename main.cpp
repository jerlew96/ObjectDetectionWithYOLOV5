#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

//常量定义
const float INPUT_WIDTH = 640.0;
const float INPUT_HIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

//文本参数
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BlUE = cv::Scalar(255,178,50);
cv::Scalar YELLOW = cv::Scalar(0,255,255);
cv::Scalar RED = cv::Scalar(0,0,255);

void draw_label(Mat& input_image, string label, int left, int top){
    //在boundingbox的最上方显示标签
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    cv::Point tlc = cv::Point(left, top);
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);

    cv::rectangle(input_image, tlc, brc, BLACK, FILLED);
    cv::putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net){
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HIGHT), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

cv::Mat post_process(cv::Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name){
    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HIGHT;
    float *data = (float *)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;

    for (int i = 0; i < rows; ++i){
        float confidence = data[4];
        //考虑检测不好的情况
        if (confidence >= CONFIDENCE_THRESHOLD){
            float * class_scores = data + 5;
            //创建一个1X85的Mat， 并且将80个类别的类别分数存储进去
            Mat scores(1, class_name.size(), CV_32FC1, class_scores);
            // 使用minMaxLoc 并且得到最好类别的分数的索引
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            //如果class score高于阈值 就继续
            if (max_class_score > SCORE_THRESHOLD){
                //将class ID和自信度都存储到之前定义的相关vector里面
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                //居中
                float cx = data[0];
                float cy = data[1];
                //bounding box的尺寸
                float w = data[2];
                float h = data[3];
                //boundingbox的坐标位置
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy-0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                //存储好的检测到box到vector里面
                boxes.push_back(Rect(left, top, width, height));

            }
        }
        data += 85;
    }
    //执行非极大抑制 并且画出预测
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++){
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        //画boudingbox
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BlUE, 3 * THICKNESS);
        //获取对应的class名称给label
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        //画类别标签
        draw_label(input_image, label, left, top);


    }
    return input_image;

}

int main(){
    //加载class list
    vector<string> class_list;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line)){
        class_list.push_back(line);
    }
    //加载图像
    Mat frame;
    frame = imread("/Users/jerlew/desktop/traffic.jpg");
    //加载模型
    Net net;
    net = readNet("/Users/jerlew/desktop/YOLOv5s.onnx");
    vector<Mat> detections;
    detections = pre_process(frame, net);
    Mat clonedframe = frame.clone();
    Mat img = post_process(clonedframe, detections, class_list);
    //放一些效率信息
    // 函数 getPerfProfile 返回inference(t) 的总时间以及每个层的计时（以 LayerTimes 为单位）
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time : %.2f ms", t);
    putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    imshow("Output", img);
    waitKey(0);
    return 0;


}

