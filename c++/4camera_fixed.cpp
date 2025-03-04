#include "NvInfer.h"  
#include<iostream>
#include <cuda_runtime.h>  
#include "NvOnnxParser.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <chrono>

#include <helper_cuda.h>  //cuda samples/common
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
using namespace cv;

// 4view ,capture
//cameraMtx, cameraDis, homographyMtx should be manullay set in cuda
// dsshow and opencv have different index. check if it matches the cameraIndex arrangement: 
// 1  2
// 3  4
// should be same type of camera (4k here), if you use low res cameras, resize should be used 
// see line 510 for using camera capture or using local images
const int cameraIndex1 = 1, cameraIndex2 = 0, cameraIndex3 = 2, cameraIndex4 = 3;
const int H = 2160, W = 3840, C = 3;      // camera , 4view images, same resolution; only C=3 is support here
//const int cutH = 1728, cutW = 3392, shift=85;      // cut each view image, cghres
const int cutH = 2160, cutW = 3840, shift = 0;      // cut each view image, cghres
const int useMap = 1;                     // 1 - use map kernel; 0 - use undistort, homography cuda kernel
const int debugMode = 0,debugFrame=1;                  // 1 - run once and save some results
const int showOpencv = 0;                 // 1 - opencv imshow; 0 - opengl imshow
bool forceBuild = false;                  // rebuild engine if engine file already exists
int gl_stop = 0;
const char* onnxPath = "onnx_old.onnx"; 
const char* enginePath = "trt_old.trt";

mutex frame1_mutex, frame2_mutex, frame3_mutex, frame4_mutex; 
condition_variable condition; 
atomic<bool> keepRunning(true); 
atomic<bool> frameReady(false); 

// cuda functions
extern "C"  void nothingCuda(unsigned char* outputDevice, unsigned char* inputDevice, int W, int H, int C);

extern "C"  void encodeImageCuda(unsigned char* encodeDevice, unsigned char* inputDevice1,
    unsigned char* inputDevice2, unsigned char* inputDevice3, unsigned char* inputDevice4,
    int W, int H, int C, int cutW, int cutH, int shift);

extern "C"  void imageToNetworkCuda(unsigned char* imageInput, float* networkInput, int W, int H, int C);

extern "C"  void networkToImageCuda(unsigned char* imageOutput, float* networkOutput, int W, int H, int C);

extern "C"  void undistortCuda(unsigned char* inputDevice, unsigned char* outputDevice, int W, int H, int C);

extern "C"  void homographyCuda(unsigned char* inputDevice, unsigned char* outputDevice, int W, int H, int C, int index);

extern "C"  void calMapCuda(float2 * mapDevice, int W, int H, int C, int cutW, int cutH, int shift);

extern "C"  void applyMapCuda(unsigned char* encodeDevice, unsigned char* inputDevice1,
    unsigned char* inputDevice2, unsigned char* inputDevice3, unsigned char* inputDevice4,
    float2 * mapDevice, int W, int H, int C, int cutW, int cutH);

extern "C"  void imageToOpenglCuda(unsigned char* imageOutput, uchar4 * openglDevice, int W, int H, int C);

class cameraHolo {
public:
    cameraHolo();

    void display();
    void initialize();
    void close();
    void captureThread(int cameraIndex, Mat& frame, mutex& frame_mutex);

private:
    //test functions
    void captureTest();

    //opengl
    GLuint textureID;  
    cudaGraphicsResource* cudaTextureResource;
    uchar4* openglDevice;
    cudaArray* texture_ptr;
    int glutWindowHandle = 0;

    //opencv
    Mat frameTest;
    Mat frameShow;  
    Mat frameCap1, frameCap2, frameCap3, frameCap4;

    //time
    int frameCounter;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::duration<double> duration;
    std::chrono::steady_clock::time_point endTime;
    float durationTime, capTime, toGPUTime, toCPUTime, kernelTime, networkTime, displayTime;
    float capTimeAverage, toGPUTimeAverage, toCPUTimeAverage, kernelTimeAverage, networkTimeAverage, displayTimeAverage;
    float fps;
    string timeStr;

    //ptr
    unsigned char* imageInput1, * imageInput2, * imageInput3, * imageInput4, *imageEncodeDevice;
    unsigned char* imageOutput;
    unsigned char* imageInputDevice1,* imageInputDevice2, * imageInputDevice3, * imageInputDevice4;
    unsigned char* imageProcessDevice1, * imageProcessDevice2,* imageProcessDevice3, * imageProcessDevice4;
    unsigned char* imageOutputDevice;
    unsigned char* imageBufferDevice; 
    float2* mapDevice;
    float* networkInputDevice;
    float* networkOutputDevice;

    //tensorrt
    ICudaEngine* engine;
    IExecutionContext* context;
    IRuntime* runtime;

    //cuda stream
    cudaStream_t streamCopy,streamKernel;

    void cghPipline();
    
    void getEngine();
    void buildEngine();
    void printNetworkInfo(INetworkDefinition* network);

    void printEngineInfo();
    void setupInference();
    void inference();

    void outText();

    void timeStart();
    void timeStop();

    void openglCreate();
    void openglDisplay();

};

cameraHolo::cameraHolo() :
    frameCap1(H,W,  CV_8UC3), frameCap2(H,W, CV_8UC3),
    frameCap3(H, W, CV_8UC3), frameCap4(H, W, CV_8UC3),
    frameShow(cutH, cutW, CV_8UC3),
    capTime(0.0f),toGPUTime(0.0f), kernelTime(0.0f), networkTime(0.0f), toCPUTime(0.0f),displayTime(0.0f),
    capTimeAverage(0.0f), toGPUTimeAverage(0.0f), kernelTimeAverage(0.0f), networkTimeAverage(0.0f),
    toCPUTimeAverage(0.0f), displayTimeAverage(0.0f),durationTime(0.0f),
    frameCounter(0), fps(0.0f){
    imageInput1 = nullptr;
    imageInput2 = nullptr;
    imageInput3 = nullptr;
    imageInput4 = nullptr;
    imageOutput = nullptr;
    imageInputDevice1 = nullptr;
    imageInputDevice2 = nullptr;
    imageInputDevice3 = nullptr;
    imageInputDevice4 = nullptr;
    imageBufferDevice = nullptr;
    imageProcessDevice1 = nullptr;
    imageProcessDevice2 = nullptr;
    imageProcessDevice3 = nullptr;
    imageProcessDevice4 = nullptr;
    imageEncodeDevice = nullptr;
    imageOutputDevice = nullptr;
    mapDevice = nullptr;
    networkInputDevice = nullptr;
    networkOutputDevice = nullptr;
}
// esc - exit; todo: use close function 
void handleKeypress(unsigned char key, int x, int y) {
    if (key == 27) {
        gl_stop = 1;
    }
}
//create opengl window and texture
void cameraHolo::openglCreate() {
    int argc = 0; char** argv = nullptr;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(cutW, cutH);
    glutWindowHandle = glutCreateWindow("CUDA OpenGL Image Display");

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cutW, cutH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glutKeyboardFunc(handleKeypress);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
 

    // Set up orthogonal projection
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glOrtho(0, cutW, cutH, 0, -1, 1);  // Define orthogonal projection matrix
    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();

    glViewport(0, 0, cutW, cutH);
}

void cameraHolo::openglDisplay() { 
    //glViewport(0, 0, cutW, cutH);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glBegin(GL_QUADS);

    //normal
    //glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    //glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    //glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    //glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);

    // reverse
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);



    glEnd();
    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();

}

void cameraHolo::initialize() {
    cout << "1, getEngine... " << endl;
    getEngine();
    cout << "complete " << endl << endl;

    cout << "2, create buffers... " << endl;
    checkCudaErrors(cudaStreamCreate(&streamCopy));
    checkCudaErrors(cudaStreamCreate(&streamKernel));
    checkCudaErrors(cudaMalloc(&imageInputDevice1, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageInputDevice2, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageInputDevice3, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageInputDevice4, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageProcessDevice1, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageProcessDevice2, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageProcessDevice3, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageProcessDevice4, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageBufferDevice, H * W * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&mapDevice, cutH * cutW * 4 * sizeof(float2)));
    checkCudaErrors(cudaMalloc(&imageOutputDevice, cutH * cutW * C * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&imageEncodeDevice, cutH * cutW * C * 4 * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&networkInputDevice, cutH * cutW * C * 4 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&networkOutputDevice, cutH * cutW * C * sizeof(float)));
    checkCudaErrors(cudaMalloc(&openglDevice , cutH * cutW * sizeof(uchar4)));

    if (useMap == 1) {
        calMapCuda(mapDevice,  W,  H,  C, cutW,  cutH, shift);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
    }

    cout << "complete " << endl << endl;

    cout << "3, setup inference... " << endl;
    setupInference();
    cout << "complete " << endl << endl;

    cout << "4, create window... " << endl;
    if (showOpencv == 1) {//opencv
        namedWindow("Display, press ESC to exit, s to save image", WINDOW_NORMAL); //WINDOW_FULLSCREEN
        resizeWindow("Display, press ESC to exit, s to save image", cutW, cutH);
    }
    else {//opengl
        openglCreate();
    }
    cout << "complete " << endl << endl;

    cout << "--------initialization complete, start display------- " << endl; 

}

void cameraHolo::close() {
    if (showOpencv) {
        destroyAllWindows();
    }
    else {
        cudaGraphicsUnregisterResource(cudaTextureResource);
        glutDestroyWindow(glutWindowHandle);
    }
    
    delete context;
    delete engine;
    delete runtime;

    checkCudaErrors(cudaStreamDestroy(streamCopy));
    checkCudaErrors(cudaStreamDestroy(streamKernel));
    checkCudaErrors(cudaFree(imageInputDevice1));
    checkCudaErrors(cudaFree(imageInputDevice2));
    checkCudaErrors(cudaFree(imageInputDevice3));
    checkCudaErrors(cudaFree(imageInputDevice4));
    checkCudaErrors(cudaFree(imageProcessDevice1));
    checkCudaErrors(cudaFree(imageProcessDevice2));
    checkCudaErrors(cudaFree(imageProcessDevice3));
    checkCudaErrors(cudaFree(imageProcessDevice4));
    checkCudaErrors(cudaFree(imageBufferDevice));
    checkCudaErrors(cudaFree(imageOutputDevice));
    checkCudaErrors(cudaFree(networkInputDevice));
    checkCudaErrors(cudaFree(networkOutputDevice));
    checkCudaErrors(cudaFree(imageEncodeDevice));
    checkCudaErrors(cudaFree(mapDevice));
    checkCudaErrors(cudaFree(openglDevice));

    cout << "--------closed------- " << endl;
}

void cameraHolo::timeStart() {
    startTime = std::chrono::high_resolution_clock::now();
}

void cameraHolo::timeStop() {
    endTime = std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    durationTime = duration.count() * 1000;  //s to ms
}
// use images to test the pipline
void cameraHolo::captureTest(){
    frameCap1 = imread("scene_0_view0.jpg");
    frameCap2 = imread("scene_0_view1.jpg");
    frameCap3 = imread("scene_0_view2.jpg");
    frameCap4 = imread("scene_0_view3.jpg");


    imageInput1 = (unsigned char*)frameCap1.data;
    imageInput2 = (unsigned char*)frameCap2.data;
    imageInput3 = (unsigned char*)frameCap3.data;
    imageInput4 = (unsigned char*)frameCap4.data;
    checkCudaErrors(cudaMemcpyAsync(imageInputDevice1, imageInput1, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
    checkCudaErrors(cudaMemcpyAsync(imageInputDevice2, imageInput2, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
    checkCudaErrors(cudaMemcpyAsync(imageInputDevice3, imageInput3, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
    checkCudaErrors(cudaMemcpyAsync(imageInputDevice4, imageInput4, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
    checkCudaErrors(cudaStreamSynchronize(streamCopy));
    nothingCuda(imageProcessDevice1, imageInputDevice1, W, H, C);
    nothingCuda(imageProcessDevice2, imageInputDevice2, W, H, C);
    nothingCuda(imageProcessDevice3, imageInputDevice3, W, H, C);
    nothingCuda(imageProcessDevice4, imageInputDevice4, W, H, C);

}
// 4 thread for camera capture
void cameraHolo::captureThread(int cameraIndex, Mat& frame, mutex& frame_mutex) {
    VideoCapture cap(cameraIndex, cv::CAP_DSHOW);  // dsshow is faster

    cap.set(CAP_PROP_FRAME_WIDTH, W);
    cap.set(CAP_PROP_FRAME_HEIGHT, H);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera " << cameraIndex << endl;
        //return;
    }
    else {
        int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH), frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
        cout << "index: " << cameraIndex << ", resolution: " << frameWidth << " x " << frameHeight << endl;
    }

    while (keepRunning) {
        if (cameraIndex==0){ timeStart(); }
        {
            lock_guard<mutex> lock(frame_mutex);
            cap.read(frame);
        }
        frameReady = true;
        condition.notify_one(); // 通知显示线程图像已经准备好
        if (cameraIndex == 0) {
            timeStop();
            capTime += durationTime;
        }
        unique_lock<mutex> lock(frame_mutex);
        condition.wait(lock, [] { return !frameReady.load(); }); // 等待图像显示完成
    }
    cap.release();
}

//CGH pipline
void cameraHolo::cghPipline() {

    //run kernel, process the image
    timeStart();
    if (useMap == 1) {
        applyMapCuda(imageEncodeDevice, imageInputDevice1, imageInputDevice2, imageInputDevice3, imageInputDevice4,
            mapDevice, W, H, C, cutW, cutH);
    }
    else {
        undistortCuda(imageInputDevice1, imageBufferDevice, W, H, C);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        homographyCuda(imageBufferDevice, imageProcessDevice1, W, H, C, 1);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        undistortCuda(imageInputDevice2, imageBufferDevice, W, H, C);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        homographyCuda(imageBufferDevice, imageProcessDevice2, W, H, C, 2);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        undistortCuda(imageInputDevice3, imageBufferDevice, W, H, C);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        homographyCuda(imageBufferDevice, imageProcessDevice3, W, H, C, 3);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        undistortCuda(imageInputDevice4, imageBufferDevice, W, H, C);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        homographyCuda(imageBufferDevice, imageProcessDevice4, W, H, C, 4);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        encodeImageCuda(imageEncodeDevice, imageProcessDevice1, imageProcessDevice2,
            imageProcessDevice3, imageProcessDevice4, W, H, C, cutW, cutH, shift);
    }
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    timeStop();
    kernelTime += durationTime;

    if (debugMode == 1 && frameCounter == debugFrame) {  // debug, save encoded image   debugMode == 1 && frameCounter == 30
        Mat frameEncode(cutH * 2, cutW * 2, CV_8UC3);
        unsigned char* imageEncodeHost = frameEncode.data; 
        checkCudaErrors(cudaMemcpyAsync(imageEncodeHost, imageEncodeDevice, cutH * cutW * 4 * C * sizeof(unsigned char), cudaMemcpyDeviceToHost, streamCopy));
        checkCudaErrors(cudaStreamSynchronize(streamCopy));

        std::string filename = "encodeImage" + std::to_string(frameCounter) + ".png";

        imwrite(filename, frameEncode);
        cout << "encodedImage saved" << endl;
    }

    //run network
    timeStart();
    inference();
    timeStop();
    networkTime += durationTime;

    // transfer. opencv: transfor to CPU. opengl: transfer to GPU 
    timeStart();
    if (showOpencv) {  //image device to opencv host
        checkCudaErrors(cudaMemcpyAsync(imageOutput, imageOutputDevice, cutH * cutW * C * sizeof(unsigned char), cudaMemcpyDeviceToHost, streamCopy));
        checkCudaErrors(cudaStreamSynchronize(streamCopy));
    }
    else {  //image device to opengl device
        imageToOpenglCuda(imageOutputDevice, openglDevice, cutW, cutH, C);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaTextureResource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
            &texture_ptr, cudaTextureResource, 0, 0));
        int size_tex_data = sizeof(GLubyte) * cutH * cutW *4;
        checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, openglDevice,
            size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTextureResource, 0)); 
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
    }
    timeStop();
    toCPUTime += durationTime;
}

//cout
void cameraHolo::outText() {

    if (frameCounter == 0) {
        capTimeAverage = capTime / 30.0f;
        toGPUTimeAverage = toGPUTime / 30.0f;
        kernelTimeAverage = kernelTime / 30.0f;
        networkTimeAverage = networkTime / 30.0f;
        toCPUTimeAverage = toCPUTime / 30.0f;
        displayTimeAverage = displayTime / 30.0f;

        capTime = 0.0f;
        toGPUTime = 0.0f;
        kernelTime = 0.0f;
        networkTime = 0.0f;
        toCPUTime = 0.0f;
        displayTime = 0.0f;
    }
    cout << "---------------- out text--------------" << endl;
    cout << "FPS: " << fps << endl;
    cout << "time for capture: " << capTimeAverage << endl;
    cout << "time for CPUtoGPU: " << toGPUTimeAverage << endl;
    cout << "time for cuda kernel: " << kernelTimeAverage << endl;
    cout << "time for network: " << networkTimeAverage << endl;
    cout << "time for GPUtoCPU: " << toCPUTimeAverage << endl;
    cout << "time for text and display: " << displayTimeAverage << endl;
    cout << "total time : " << capTimeAverage + toGPUTimeAverage + kernelTimeAverage
        + networkTimeAverage + toCPUTimeAverage + displayTimeAverage << endl;

}

void cameraHolo::display() {
    // FPS
    std::chrono::steady_clock::time_point startTimeTotal = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationTotal;
    std::chrono::steady_clock::time_point endTimeTotal;


    captureTest(); //Read local images for testing. Comment out this section and uncomment the following two sections to use the camera for live capture
    
    //uncomment for camera capture
    /*thread cam1_thread(&cameraHolo::captureThread, this, cameraIndex1, std::ref(frameCap1), std::ref(frame1_mutex));
    thread cam2_thread(&cameraHolo::captureThread, this, cameraIndex2, std::ref(frameCap2), std::ref(frame2_mutex));
    thread cam3_thread(&cameraHolo::captureThread, this, cameraIndex3, std::ref(frameCap3), std::ref(frame3_mutex));
    thread cam4_thread(&cameraHolo::captureThread, this, cameraIndex4, std::ref(frameCap4), std::ref(frame4_mutex));*/

    imageInput1 = (unsigned char*)frameCap1.data;
    imageInput2 = (unsigned char*)frameCap2.data;
    imageInput3 = (unsigned char*)frameCap3.data;
    imageInput4 = (unsigned char*)frameCap4.data;

    imageOutput = (unsigned char*)frameShow.data;

    char key = 0;
    timeStart();
    while (true) {
        frameCounter++;
        if (frameCounter  == 31) {
            endTimeTotal = std::chrono::high_resolution_clock::now();
            durationTotal = endTimeTotal - startTimeTotal;
            fps = frameCounter / durationTotal.count();
            startTimeTotal = endTimeTotal;
            frameCounter = 0;
            outText();
        }
 
        {
            //uncomment for camera capture
            //unique_lock<mutex> lock1(frame1_mutex);
            //unique_lock<mutex> lock2(frame2_mutex);
            //unique_lock<mutex> lock3(frame3_mutex);
            //unique_lock<mutex> lock4(frame4_mutex);
            //condition.wait(lock1, [] { return frameReady.load(); }); 
            timeStop();
            capTime += durationTime;


            // transfer to GPU
            timeStart();
            imageInput1 = (unsigned char*)frameCap1.data;
            imageInput2 = (unsigned char*)frameCap2.data;
            imageInput3 = (unsigned char*)frameCap3.data;
            imageInput4 = (unsigned char*)frameCap4.data;

            checkCudaErrors(cudaMemcpyAsync(imageInputDevice1, imageInput1, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
            checkCudaErrors(cudaMemcpyAsync(imageInputDevice2, imageInput2, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
            checkCudaErrors(cudaMemcpyAsync(imageInputDevice3, imageInput3, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
            checkCudaErrors(cudaMemcpyAsync(imageInputDevice4, imageInput4, H * W * C * sizeof(unsigned char), cudaMemcpyHostToDevice, streamCopy));
            checkCudaErrors(cudaStreamSynchronize(streamCopy));
            timeStop();
            toGPUTime += durationTime;
        }
        

        cghPipline();  

        timeStart();
        if (showOpencv == 1) {
            //addText();
            imshow("Display, press ESC to exit, s to save image", frameShow);
            if (debugMode == 1 &&frameCounter == debugFrame) {
                //imageOutput = (unsigned char*)frameShow.data;
                string filename = "frameShow.png";
                imwrite("frameShow.png", frameShow);
                cout << "Saved image as " << filename << endl;
                cout << "exit because debugMode " << endl;
                //flip(frameCap1, frameCap1,-1);
                //flip(frameCap2, frameCap2, -1);
                imwrite("1.png", frameCap1);
                imwrite("2.png", frameCap2);
                imwrite("3.png", frameCap3);
                imwrite("4.png", frameCap4);

                break;
            }
            key = waitKey(1);
            if (key == 's' || key == 'S') {
                string filename = "captured_image.png";
                imwrite("captured_image.png", frameShow);
                cout << "Saved image as " << filename << endl;
            }
            else if (key == 27) { // ASCII 27 -- ESC
                cout << "exit because ESC is pressed " << endl;
                break;
            }
        }  
        else {
            openglDisplay();
            if (debugMode == 1 && frameCounter == debugFrame) {
                break;
            }
            if (gl_stop == 1) {
                break;
            }
        }
        timeStop();
        displayTime += durationTime;

        frameReady = false;
        condition.notify_all(); 
        timeStart();
    }
}
//----------------------------------------------------------------------
//---------------------------tensorrt-----------------------------------
//----------------------------------------------------------------------
class Logger :public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity < Severity::kWARNING)
            std::cout << msg << std::endl;
    };
}logger;

void cameraHolo::printNetworkInfo(INetworkDefinition* network) {
    cout << "-------printNetworkInfo------- " << std::endl;
    auto inputNum = network->getNbInputs();// assume 1
    nvinfer1::Dims inDims = network->getInput(0)->getDimensions();// assume 4
    auto inputDim = inDims.nbDims;
    assert(inputNum == 1);
    assert(inputDim == 4);
    cout << "input num: " << inputNum << std::endl;
    cout << "input dim: " << inputDim << std::endl;
    cout << "input size: " << inDims.d[0] << "," << inDims.d[1] << ","
        << inDims.d[2] << "," << inDims.d[3] << std::endl;

    auto outputNum = network->getNbOutputs();// assume 1
    nvinfer1::Dims outDims = network->getOutput(0)->getDimensions();// assume 4
    auto outputDim = outDims.nbDims;
    assert(outputNum == 1);
    assert(outputDim == 4);
    cout << "output num: " << outputNum << std::endl;
    cout << "output dim: " << outputDim << std::endl;
    cout << "output size: " << outDims.d[0] << "," << outDims.d[1] << ","
        << outDims.d[2] << "," << outDims.d[3] << std::endl;
}

//build tensorrt network from onnx, and save
void cameraHolo::buildEngine() {
    IBuilder* builder = createInferBuilder(logger);
    assert(builder != nullptr);

    INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);

    IParser* parser = createParser(*network, logger);
    assert(parser != nullptr);

    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);

    // set config
    config->setFlag(BuilderFlag::kFP16);
    //config->setMaxWorkspaceSize(100);
    //config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 128);
    //config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);

    cout << "Loading ONNX file from: " << onnxPath << endl;
    if (parser->parseFromFile(onnxPath, static_cast<int32_t>(ILogger::Severity::kWARNING))) {
        cout << "ONNX parsed successfully" << endl;
    }
    else {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << parser->getError(i)->desc() << std::endl;
        }
        exit(1);
    }

    cout << "converting onnx to  engine, this should take a while........ " << endl;
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    assert(serializedModel != nullptr);
    cout << "successfully  convert onnx to  engine " << std::endl;

    std::ofstream a(enginePath, std::ios::binary);
    a.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    cout << "engine saved: " << enginePath << endl;

    printNetworkInfo(network);

    delete serializedModel;
    delete parser;
    delete config;
    delete network;
    delete builder;
   
}

void cameraHolo::getEngine() {
    cout << "-------getEngine info------- " << std::endl;
    if (forceBuild) {
        cout << "usd force_build, build engine..." << endl;
        buildEngine();
        return;
    }
    else {
        std::ifstream file(enginePath);
        if (file.good()) {
            cout << "engine exists, will use existing engine..." << endl;
            return;
        }
        else {
            cout << "engine doesn't exist, build engine..." << endl;
            buildEngine();
            return;
        }
    }
}

void cameraHolo::printEngineInfo()
{
    cout << "-------printEngineInfo------- " << std::endl;
    auto numIOTensors = engine->getNbIOTensors();
    cout << "numbers of the IOTensors: " << numIOTensors << endl;
    assert(numIOTensors == 2);  //in out

    auto inputName = engine->getIOTensorName(0);
    auto inputShape = engine->getTensorShape(inputName);
    cout << "inputname: " << inputName << std::endl;;
    auto inputDim = inputShape.nbDims;
    assert(inputDim == 4);
    cout << "input dim: " << inputDim << std::endl;
    cout << "input size: " << inputShape.d[0] << "," << inputShape.d[1]
        << "," << inputShape.d[2] << "," << inputShape.d[3] << std::endl;

    auto outputName = engine->getIOTensorName(1);
    auto outputShape = engine->getTensorShape(outputName);
    cout << "outputname: " << outputName << std::endl;;
    auto outputDim = outputShape.nbDims;
    assert(outputDim == 4);
    cout << "output dim: " << outputDim << std::endl;
    cout << "output size: " << outputShape.d[0] << "," << outputShape.d[1]
        << "," << outputShape.d[2] << "," << outputShape.d[3] << std::endl;
}

void cameraHolo::setupInference() {
    char* readEngine = nullptr;
    size_t size = 0;
    std::ifstream file(enginePath, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        readEngine = new char[size];
        file.read(readEngine, size);
        file.close();
        cout << "engine file read" << endl;
    }
    else {
        cout << "enging file is not good" << endl;
        exit(1);
    }

    runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(readEngine, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] readEngine;

    printEngineInfo();

    bool status;
    status = context->setTensorAddress("input", networkInputDevice);
    assert(status != false);
    status = context->setTensorAddress("output", networkOutputDevice);
    assert(status != false);

}

void cameraHolo::inference()
{
    imageToNetworkCuda(imageEncodeDevice, networkInputDevice,  cutW*2,  cutH*2,  C);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    bool status = context->enqueueV3(streamKernel);
    assert(status != false);
    checkCudaErrors(cudaStreamSynchronize(streamKernel));

    networkToImageCuda(imageOutputDevice, networkOutputDevice,cutW, cutH, C);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
}

int main() {
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    cameraHolo cameraHoloRun;
    cameraHoloRun.initialize();
    cameraHoloRun.display();
    cameraHoloRun.close();
    return 0;
}