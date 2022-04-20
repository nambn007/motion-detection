#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudaarithm.hpp>
#include <unistd.h>

#define DEFAULT_DELAY_FRAME_COUNT 3
#define DEFAULT_THRESH_PIXEL 25
#define DEFAULT_MIN_SIZE_FOR_MOVEMENT 4000
#define DEFAULT_MOTION_DETECTED_PERSISTANCE 5

using namespace cv;
using namespace std;
 
int main(int argc, char **argv) {
    
    CommandLineParser cmd(argc, argv, 
            "{ mode      | 1 | 0 - camera, 1 - video}"
            "{ v video   | /home/namdz/vinbigdata/data/dreamcity_lpr/test2.mp4 | specify input video}"
            "{ g gpu     | true | GPU or CPU}"
            "{ threshold | 5.0 | Threshold for magnitude}"
            "{ threshold_pixel | 200 | Threshold for number of pixel for detect motion or not}"
            );

    cmd.about("Farneback's optical flow samples.");
    bool gpuMode = cmd.get<float>("gpu");
    int mode     = cmd.get<int>("mode");
    string pathVideo = cmd.get<string>("video");
    int THRESHOLD_PIXEL = cmd.get<int>("threshold_pixel");
    
    cout << "Computer vision" << endl;
    cout << "Mode: " << std::to_string(mode) << endl;
    
    VideoCapture cap;
    if(mode == 0){
        cap.open(0); // Read from camera;
    }else{
        if(pathVideo.empty()){
            cerr << "Path video cannot empty when mode = 1\n";
            return -1;
        }
        cap.open(pathVideo);
    }
    if(!cap.isOpened()){
        cerr << "Cannot open camera or video\n";
        return -1;
    }

    Mat frame, fgMask, frame_resize;
    Ptr<BackgroundSubtractorMOG2> pBackSub = createBackgroundSubtractorMOG2();
    int rescale = 3, nonzero_pixel = -1;
    bool moiton_detected = true;
    while(true){
        cap.read(frame);
        if(frame.empty()) break;
        
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        resize(frame, frame_resize, Size(frame.cols / rescale, frame.rows / rescale), 0, 0, INTER_LINEAR);
        pBackSub->apply(frame_resize, fgMask);
        
        // Erode and Dialte 
        erode(fgMask, fgMask, Mat());
        dilate(fgMask ,fgMask , Mat());
        
        // Count non zeropixel
        nonzero_pixel = countNonZero(fgMask);
        if(nonzero_pixel < THRESHOLD_PIXEL){
            moiton_detected = false;
        } else moiton_detected = true;

        string result =  moiton_detected ? "Motion detected" : "Non Motion";
        putText(fgMask, result, Point(30, 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 1);
        imshow("Fgmask", fgMask);

        if(waitKey(1) == 27){
            break;
        }
    }

    return 0;
}