#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
//#include <darknet.h>


using namespace cv;
using namespace std;
using namespace dnn;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image
vector <string> classes;

//struct 


void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    string label = format("%.2f", conf);
    //cout << label;
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
        //cout << classes[classId] << '\n';
        if (classes[classId] == "person" || classes[classId] == "car")
        {
            rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            top = max(top, labelSize.height);
            rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
            putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
        }
    }
    //cout << label;

    /*int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);*/
}

void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                //cout << classIdPoint.x << '\n';
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    /*for (size_t i = 0; i < indices.size(); i++)
    {
        cout << indices[i] << '\n';
    }*/
    for (size_t i = 0; i < classIds.size(); i++)
    {
        cout << classIds[i] << '\n';
    }

    cout << "Frame done!\n";

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}


vector <string> getOutputsNames(const Net& net)
{
    //FILE* layersfile;

    //layersfile = fopen("out.txt", "w");

    //Net net_fix = net;

    static vector<String> names;
    if (names.empty())
    {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        //Ptr <dnn::Layer> strLayernames;
        //vector <Ptr <dnn::Layer>> StrLayers;
        for (size_t i = 0; i < outLayers.size(); ++i)
        {
            names[i] = layersNames[outLayers[i] - 1];
        }

        /*for (size_t i = 0; i < names.size(); i++)
        {
            cout << names[i] << '\n';
        }*/
        /*for (size_t i = 0; i < layersNames.size(); i++)
        {
            //cout << layersNames[i] << '\n';
            strLayernames = net_fix.getLayer(net_fix.getLayerId(layersNames[i]));
            StrLayers = net_fix.getLayerInputs(net_fix.getLayerId(layersNames[i]));
            //strLayernames->getDefaultName();
            //strLayernames->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));
            //cout << strLayernames << '\n';

            //fprintf(layersfile, "%s\n", layersNames[i]);
        }*/
        /*for (size_t i = 0; i < StrLayers.size(); i++)
        {
            cout << StrLayers[i] << '\n';
        }*/
        /*for (size_t i = 0; i < outLayers.size(); i++)
        {
            cout << outLayers[i] << '\n';
        }*/
    }

    //fclose(layersfile);

    return names;
}


int main()
{
    string modelConfiguration = "yolov4.cfg";
    string modelWeights = "yolov4.weights";
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;

    //detection a;

    while (getline(ifs, line))
    {
        classes.push_back(line);
    }

    string inputfile = "C:\\Users\\Бобур Ибрагимов\\source\\repos\\Tracking object\\videoplayback.mp4";
    VideoWriter video;
    VideoCapture cap("videoplayback.mp4");

    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat frame, blob;

    string outputfile = "C:\\Users\\Бобур Ибрагимов\\source\\repos\\Tracking object\\video2.avi";

    video.open(outputfile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));


    static const string kwinName = "Object detection";
    namedWindow(kwinName, WINDOW_NORMAL);



    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputfile << endl;
            //waitKey(300);
            break;
        }

        blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        net.setInput(blob);

        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        postprocess(frame, outs);

        /*for (int i = 0; i < outs.size(); i++)
        {
            cout << outs[i] << '\n';
        }*/


        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);


        imshow(kwinName, frame);
    }

    

    cap.release();
    video.release();

    return 0;
}

