#include <c++/15.2.0/filesystem>
#include <c++/15.2.0/future>
#include <c++/15.2.0/iostream>
#include <c++/15.2.0/memory>
#include <c++/15.2.0/mutex>
#include <c++/15.2.0/queue>
#include <c++/15.2.0/vector>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

/*template <typename T> class Queue{
    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable condition_variable_;
    
    public:
        void push(T _){
            {
                std::lock_guard<std::mutex>> lock(mutex_);
                queue_.push(std::move(_));
            }
            condition_variable_.notify_all();
        }
        T pop(){
            std::unique_lock<std::mutex>> lock(mutex_);
            condition_variable_.wait(lock, [this]{return !queue.empty();});
            T _ = std::move(queue_.front());
            queue_.pop();
            return _;
        }
        bool empty(){
            std::lock_guard<std::mutex>> lock(mutex_);
            return queue_.empty();
        }

}*/
std::vector<cv::Point2f> openpose_classification(cv::dnn::Net &net, cv::Mat &image);
std::vector<float> normalize_bounding_box(std::vector<cv::Point2f> &points);
std::vector<float> normalize_euclidean(std::vector<cv::Point2f> &points);
std::vector<cv::Point2f> mediapipe_classification(cv::dnn::Net &classifier, cv::Mat &image);
void support_vector_machine(cv::dnn::Net &net, std::string directory);

void display_frame(cv::dnn::Net &net, cv::Ptr<cv::ml::SVM> model);

int main(int argc, char* argv[]){
    std::cout << "Enter" << std::endl;
    cv::dnn::Net net = cv::dnn::readNetFromCaffe("pose_deploy.prototxt", "pose_iter_102000.caffemodel");
    cv::dnn::Net mediapipe_landmark = cv::dnn::readNetFromONNX("palm_detection_mediapipe_2023feb.onnx");
    cv::dnn::Net mediapipe_classifier = cv::dnn::readNetFromONNX("handpose_estimation_mediapipe_2023feb.onnx");
    //support_vector_machine(mediapipe_classifier, "F:\\ARAP Deformer\\SignAlphaSet");
    //support_vector_machine(mediapipe_classifier, "F:\\Data\\American Sign Language Mediapipe Landmark Dataset");
    //support_vector_machine(mediapipe_classifier, "F:\\Data\\American Sign Language Hand Skeleton Dataset");
    /*cv::Ptr<cv::ml::SVM> openpose_model = cv::ml::SVM::load("openpipe_model.xml");
    if(openpose_model.empty()){
        std::cerr << "Error: Could not load the SVM model!" << std::endl;
        return 0;
    }*/
    cv::Ptr<cv::ml::SVM> mediapipe_model = cv::ml::SVM::load("demo_model.xml"); // Use demo_model.xml for maximum accuracy and result as it was built from a larger dataset(SignAlphaSet)
    if(mediapipe_model.empty()){
        std::cerr << "Error loading the model" << std::endl;
        return 0;
    }
    display_frame(mediapipe_classifier, mediapipe_model);
    
    /*
    //std::shared_ptr<cv:Mat> frame = std::make_shared<cv::Mat>(bgr);
    Queue<std::shared_ptr<cv::Mat>> instance;
    std::vector<std::thread>> thread_(3);
    thread_[0] = std::thread(capture_frame, frame);
    thread_[1] = std::thread(capture_frame, frame);
    thread_[2] = std::thread(handler, frame);
    short spawn = 3, _ = 0;
    while(spawn > 0){
        if(thread_[_].joinable()){
            thread_[_].join();
            spawn--;
        }
        _ == 2 ? _ = 0 : ++_;
    }*/
    return 1;
}

void display_frame(cv::dnn::Net &net, cv::Ptr<cv::ml::SVM> model){
    cv::VideoCapture video_capture(1, cv::CAP_ANY);
    if(!video_capture.isOpened()){
        std::cerr << "Error! Unable to open camera" << std::endl;
        return;
    }
    cv::Mat bgr;
    std::string alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY", translated_text = "";
    while(1){
        video_capture >> bgr;
        if(bgr.empty()){
            std::cerr << "Blank frame grabbed" << std::endl;
            break;
        }

        /*//openpose
        std::vector<cv::Point2f> landmark = openpose_classification(net, bgr);
        if(landmark.size() == 21){
            std::vector<float> features = normalize_bounding_box(landmark);
            float response = model->predict(cv::Mat(features).t());
            char predictedLetter = (char)('A' + (int)response);
            if(predictedLetter >= 'J') predictedLetter++; // Skip 'J' mapping if you didn't train it
            std::string label = "Letter: ";
            label += predictedLetter;
            cv::putText(bgr, label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
        }*/

        //mediapipe
        std::vector<cv::Point2f> landmark = mediapipe_classification(net, bgr);
        int key = cv::waitKey(1);
        if(landmark.size() == 21){
            std::vector<float> features = normalize_bounding_box(landmark);
            cv::Mat sample = cv::Mat(features).t();
            sample.convertTo(sample, CV_32F);
            float response = model->predict(sample);
            char letter = alphabet[(int)response];
            std::string check(1, letter);
            cv::putText(bgr, check, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
            if (key == 32) { 
                translated_text += letter; 
            }
            //translated_text += letter;
            cv::putText(bgr, translated_text, cv::Point(30, bgr.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
        }
        
        cv::imshow("Live", bgr);
        if (cv::waitKey(1) == 27) break;
        if (key  == 8 || key  == 127){
            if (!translated_text.empty()) translated_text.pop_back();
        }
    }
    return;
}

std::vector<cv::Point2f> openpose_classification(cv::dnn::Net &net, cv::Mat &image){
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(368, 368), cv::Scalar(0, 0, 0), 0, 0);
    net.setInput(blob);
    cv::Mat output = net.forward();
    uint32_t height = output.size[2], width = output.size[3], nPoints = 21;
    std::vector<cv::Point2f> points(nPoints);
    for(short _ = 0; _ < nPoints; _++){
        cv::Mat map(height, width, CV_32F, output.ptr(0, _));
        cv::Point2f p(-1,-1);
        cv::Point maxLoc;
        double prob;
        cv::minMaxLoc(map, NULL, &prob, NULL, &maxLoc);
        if(prob > 0.1) {
            p = maxLoc;
            float x = (maxLoc.x * image.cols) / width, y = (maxLoc.y * image.rows) / height;
            cv::circle(image, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
        }
        points[_] = p;
    }
    return points;
}

std::vector<float> normalize_bounding_box(std::vector<cv::Point2f> &points){
    cv::Point2f wrist = points[0];
    std::vector<float> data;
    float maxDist = 0.0f;
    for(cv::Point2f &p : points){
        cv::Point2f normalized = p - wrist;
        data.push_back(normalized.x);
        data.push_back(normalized.y);
        maxDist = std::max({maxDist, std::abs(normalized.x), std::abs(normalized.y)});
    }
    for(float &val : data) val /= maxDist;
    return data;
}

std::vector<float> normalize_euclidean(std::vector<cv::Point2f> &points){
    cv::Point2f wrist = points[0];
    std::vector<float> data;
    float distance = 0.0f;
    for(cv::Point2f &p : points){
        cv::Point2f delta = p - wrist;
        float d = std::sqrt(delta.x * delta.x + delta.y * delta.y);
        if(d > distance) distance = d;
        data.push_back(delta.x);
        data.push_back(delta.y);
    }
    if(distance > 1e-6) for(float &val : data) val /= distance;
    return data;
}

std::vector<cv::Point2f> mediapipe_classification(cv::dnn::Net &classifier, cv::Mat &image){
    std::vector<int> inputLayerIds = classifier.getUnconnectedOutLayers();
    std::vector<std::vector<int>> in_layer, out_layer;
    classifier.getLayerShapes(cv::dnn::MatShape(), 0, in_layer, out_layer);
    std::vector<int> shape = in_layer[0];
    if(shape[1] == 3){
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0/255.0, cv::Size(shape[3], shape[2]), cv::Scalar(0,0,0), 1, 0);
        classifier.setInput(blob);
    } else{    
        cv::Mat resized, rgb, floatImage;
        cv::resize(image, resized, cv::Size(224, 224));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(floatImage, CV_32F, 1.0 / 255.0);
        cv::Mat blob(4, shape.data(), CV_32F, floatImage.data);
        classifier.setInput(blob.clone());
    }
    std::vector<cv::Mat> output;
    classifier.forward(output, classifier.getUnconnectedOutLayersNames());
    float *data = (float*)output[0].data;
    std::vector<cv::Point2f> points;
    for(short i = 0; i < 21; ++i){
        float x = (data[i * 3] / 224.0) * image.cols, y = (data[i * 3 + 1] / 224.0) * image.rows;
        points.push_back(cv::Point2f(x, y));
        cv::circle(image, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
    }

    const int POSE_PAIRS[][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},         
    {0, 5}, {5, 6}, {6, 7}, {7, 8},         
    {5, 9}, {9, 10}, {10, 11}, {11, 12},    
    {9, 13}, {13, 14}, {14, 15}, {15, 16},  
    {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20}
};
    for (int n = 0; n < 21; n++){
        cv::Point2f partA = points[POSE_PAIRS[n][0]];
        cv::Point2f partB = points[POSE_PAIRS[n][1]];
 
    if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
        continue;
 
    cv::line(image, partA, partB, cv::Scalar(0,255,255), 2);
    //circle(image, partA, 8, Scalar(0,0,255), -1);
    //circle(image, partB, 8, Scalar(0,0,255), -1);
}
    return points;
}

void support_vector_machine(cv::dnn::Net &net, std::string directory){
    cv::Mat data, label;
    std::string alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY";
    for(uint8_t _ = 0; _ < alphabet.size(); _++){
        std::filesystem::path letter_directory = std::filesystem::path(directory) / std::string(1, alphabet[_]) / "*.jpg";
        std::string letter = letter_directory.string();
        std::vector<std::string> filePaths;
        cv::glob(letter, filePaths);
        std::cout << "Processing letter: " << alphabet[_] << std::endl;
        for(const std::string& path : filePaths){
            cv::Mat image = cv::imread(path);
            if(image.empty()) continue;
            std::vector<cv::Point2f> landmark = mediapipe_classification(net, image);
            if(landmark.size() == 21){
                std::vector<float> normalized_landmark = normalize_bounding_box(landmark);
                data.push_back(cv::Mat(normalized_landmark).t());
                label.push_back(_);
            } else{
                std::cout << "Invalid"<< std::endl;
            }
        }
    }
    label.convertTo(label, CV_32S);
    data.convertTo(data, CV_32F);
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setGamma(0.1);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(data, cv::ml::ROW_SAMPLE, label);
    svm->save("demo_model.xml");
    return;
}

//std::cout << "Processed Letter: " << alphabet[_] << " | Image: " << path << std::endl;

/*void handler(std::shared_ptr<cv:Mat> frame){
    bool task = 0;
    cv::Mat bgr = frame, ycbcr, mask, target, otsu, binary, trace;
    cv::cvtColor(bgr_, ycbcr, cv::COLOR_BGR2YCrCb);
    cv::Scalar lower_skin(0, 133, 77);
    cv::Scalar upper_skin(255, 173, 127);
    cv::inRange(ycbcr, lower_skin, upper_skin, mask);
    cv::bitwise_and(bgr_, bgr_, target, mask);
    std::vector<cv::Mat> channel;
    cv::extractChannel(ycbcr, otsu, 1);
    double threshold = cv::threshold(channel[1], otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::bitwise_and(mask, otsu, binary);
    cv::GaussianBlur(binary, binary, cv::Size(5, 5), 0);
    std::vector<std::vector<cv::Point>> contour;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contour, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> hull(contour.size());
    std::vector<cv::Point> approx;
    struct HandCandidate{
        std::vector<cv::Point> contour;
        int x;
    };
        std::vector<HandCandidate> validHands;
        for(size_t _ = 0; _ < contour.size(); _++){
            if(cv::convexHull(contour[_], hull[_]) < 1000) continue;
            cv::approxPolyDP(contour[_], approx, 0.01 * cv::arcLength(contour[_], 1), 1);
            if(cv::matchShapes(countour[_], template, cv::CONTOURS_MATCH_I1, 0) < 0.15){
                validHands.push_back({contour[_], cv::boundingRect(contour[_]).x});
                //Match
            }
        }
        std::sort(validHands.begin(), validHands.end(), [](const HandCandidate& a, const HandCandidate& b){return a.x < b.x;});
        for(size_t i = 0; i < validHands.size(); i++) {
            cv::Rect box = cv::boundingRect(validHands[i].contour);
            std::string label = (i == 0) ? "Left Hand" : "Right Hand";
            cv::drawContours(bgr_, std::vector<std::vector<cv::Point>>{validHands[i].contour}, -1, cv::Scalar(0, 255, 0), 2);
            cv::putText(bgr_, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
        return bgr_;
    
    std::future<bool> task_handler;
    while(1){
        cap >> bgr;
        if(bgr.empty()){
            std::cerr << "ERROR! blank frame grabbed" << std::endl;
            break;
        }
        if(!task){
            task_handler = std::async(std::launch::async, handler, bgr.clone());
            task = 1;
        } else if(task_handler.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready){
            cv::Mat task_ = task_handler.get();
            task = 0;
        }
        cv::imshow("Live", bgr);
        if (cv::waitKey(1) == 27) break;
    }
}*/
//cv::flip(bgr, bgr, 1); // 1 = flip horizontally