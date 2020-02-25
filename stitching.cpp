#include <iostream>
#include <thread>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"

const std::string keys = {
    "{help h usage ? |      | print this message       }"
    "{@image1 i      |      | Top image for compare    }"
    "{@image2 j      |      | Bottom image for compare }"
    "{@output        |      | name of output image     }"
    "{threads t      |1     | number of threads to use }"
};

const int maxThreads = std::thread::hardware_concurrency();

uint64_t compareRows(const cv::Mat &top, const cv::Mat &bot,
                   const std::size_t offset)
{
    uint64_t difference = 0.0;
    cv::Mat topSection = top(cv::Range(0, top.rows), cv::Range(offset, top.cols));
    topSection.forEach<uint8_t>(
        [bot, &difference](uint8_t &pixel, const int32_t *pos) {
            difference += abs(pixel - bot.ptr<uint8_t>(pos[0])[pos[1]]);
        }
    );
    return difference;
}

void combine(const cv::Mat &top, const cv::Mat &bot,
             const std::size_t topIdx, const std::size_t offset,
             cv::Mat &output)
{
    const std::size_t outCols = top.cols >= bot.cols ? top.cols : bot.cols;
    output = cv::Mat::zeros(topIdx + bot.rows, outCols, CV_8U);

    // Copy from 0 to the first overlapping index from top image
    top(cv::Range(0, topIdx),
        cv::Range(0, top.cols)).forEach<uint8_t>(
            [&output](uint8_t &pixel, const int32_t *pos) {
                output.ptr<uint8_t>(pos[0])[pos[1]] = pixel;
            });

    // Then use bottom for the rest
    bot(cv::Range(0, bot.rows),
        cv::Range(offset, bot.cols - offset)).forEach<uint8_t>(
            [topIdx, offset, &output](uint8_t &pixel, const int32_t *pos) {
                output.ptr<uint8_t>(pos[0] + topIdx)[pos[1] + offset] = pixel;
            });
}

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Vertical Image Stitching v1.0.0");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::string imagePath1 = parser.get<std::string>("@image1");
    std::string imagePath2 = parser.get<std::string>("@image2");
    std::string outputPath = parser.get<std::string>("@output");

    cv::Mat top = cv::imread(imagePath1, cv::IMREAD_GRAYSCALE);
    if (top.rows < 1 || top.cols < 1) {
        std::cerr << "Failed to open top image" << std::endl;
        return 1;
    }
    cv::Mat bot = cv::imread(imagePath2, cv::IMREAD_GRAYSCALE);
    if (bot.rows < 1 || bot.cols < 1) {
        std::cerr << "Failed to open bot image" << std::endl;
        return 1;
    }

    // Set number of threads for opencv to use
    int numThreads = parser.get<int>("threads");
    if (numThreads > maxThreads) {
        numThreads = maxThreads;
    }
    cv::setNumThreads(numThreads);

    std::size_t offset = 0;
    std::size_t bestFitTopIdx = 0;
    // Set default to absurd value so anything will match better
    uint64_t bestFit = std::numeric_limits<uint64_t>::max();

    // Use the smallest column count and "wiggle" back and forth
    const std::size_t minColSize = top.cols >= bot.cols ? bot.cols : top.cols;

    // TODO: Since we're working from the bottom up, establish a certainty
    // threshold. When bestFit <= threshold, terminate loop
    for (std::size_t i = top.rows - 1; i > 0; i--) {
        for (std::size_t j = 0; j < (top.cols - minColSize + 1); j++) {
            cv::Mat topSection = top(cv::Range(i, i+1), cv::Range(0, top.cols));
            cv::Mat botSection = bot(cv::Range(0, 1), cv::Range(0, minColSize));
            uint64_t tempFit = compareRows(topSection, botSection, j);
            if (tempFit < bestFit) {
                bestFit = tempFit;
                bestFitTopIdx = i;
                offset = j;
            }
        }
    }
    std::cout << "Found best fit at index " << bestFitTopIdx
        << " of top image" << std::endl;

    cv::Mat output;
    combine(top, bot, bestFitTopIdx, offset, output);

    cv::imwrite(outputPath, output);
    std::cout << "Wrote results to file " << outputPath << std::endl;

    return 0;
}