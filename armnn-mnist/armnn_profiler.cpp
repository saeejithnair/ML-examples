/*
 * License Information
 */

#include <iostream>
#include "armnn/ArmNN.hpp"
#include <cstring>

#include <fstream>
#include <sstream>

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>

#include <opencv2/dnn.hpp>
#include "armnn_cv.hpp"

#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfParser/ITfParser.hpp"

#include "mnist_loader.hpp"

/* 
 * This application profiles the speed and accuracy of using ArmNN vs. Open-CV for object detection.
 */

/*static void help()
{
  cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
       "-- pause video [p] and draw a bounding box around the target to start the tracker\n"
       "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
       "Call:\n"
       "./tracker <tracker_algorithm> <video_name> <start_frame> [<bounding_frame>]\n"
       "tracker_algorithm can be: MIL, BOOSTING, MEDIANFLOW, TLD, KCF, GOTURN, MOSSE.\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
}*/
	
std::vector<float> ArmnnBlobFromImage(ArmnnImage& image)
{
	/*if (newWidth != image.GetWidth() || newHeight != image.GetHeight())
    {
        image.Resize(newWidth, newHeight, CHECK_LOCATION());
    }*/
    
    return GetImageDataInArmNnLayoutAsNormalizedFloats(ImageChannelLayout::Rgb, image);
}

armnn::IRuntimePtr ArmnnReadNetFromTf( const char* model_path, armnn::NetworkId& networkIdentifier,
												armnnTfParser::BindingPointInfo& inputBindingInfo,
												armnnTfParser::BindingPointInfo& outputBindingInfo)
{
	// Import the TensorFlow model. Note: use CreateNetworkFromBinaryFile for .pb files.
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(model_path,
                                                                   { {"Placeholder", {1, 784, 1, 1}} },
                                                                   { "Softmax" });

    // Find the binding points for the input and output nodes
    inputBindingInfo = parser->GetNetworkInputBindingInfo("Placeholder");
    outputBindingInfo = parser->GetNetworkOutputBindingInfo("Softmax");

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, {armnn::Compute::CpuRef}, runtime->GetDeviceSpec());
    
    // Load the optimized network onto the runtime device
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    return runtime;
}

/*armnn::IRuntimePtr ArmnnSetInput()
{
	

    return ;
}

armnn::Status ArmnnForwardPass(armnn::IRuntimePtr runtime, )
{

}*/

const cv::String keys = {
    "{help h usage ? |      | help message         }"
    "{@model_path    |      | path to the model    }"
    "{@image_path    |      | path to image        }"
    "{@config_path   |      | path to config file  }"
};

int main(int argc, char** argv)
{
    using namespace armnn;

    cv::CommandLineParser clparser( argc, argv, keys );

    /* Read inputs from command line */
    cv::String model_path = clparser.get<cv::String>(0);
    cv::String image_path = clparser.get<cv::String>(1);
    cv::String config_path = clparser.get<cv::String>(2);

    /*std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string config_path = argv[3];*/

    /* Convert image path to ArmNN image object */
    ArmnnImage armnn_image(image_path.c_str());

    /* Resize/normalize image into ArmNN "blob" */
    // Container to hold normalized image data.
    std::vector<float> armnn_blob = ArmnnBlobFromImage(armnn_image);

    armnnTfParser::BindingPointInfo inputBindingInfo;
    armnnTfParser::BindingPointInfo outputBindingInfo;
    armnn::NetworkId networkIdentifier;

    armnn::IRuntimePtr runtime = ArmnnReadNetFromTf(model_path.c_str(), networkIdentifier, 
                                                    inputBindingInfo, outputBindingInfo);

    // Run a single inference on the test image
    std::array<float, 10> output;
    int64 frameTime = cv::getTickCount();
    armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
                                                 MakeInputTensors(inputBindingInfo, &armnn_blob[0]),
                                                 MakeOutputTensors(outputBindingInfo, &output[0]));
    frameTime = cv::getTickCount() - frameTime;
    
    // Convert 1-hot output to an integer label and print
    int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::cout << "Predicted: " << label << std::endl;
    //std::cout << "   Actual: " << input->label << std::endl;
    return 0;
}