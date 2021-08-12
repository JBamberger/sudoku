#include <SudokuDetector.h>

#include <config.h>
#include <drawutil.h>
#include <utils.h>

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

class SudokuApplication
{
    std::unique_ptr<SudokuDetector> detector;

  public:
    explicit SudokuApplication(const std::string& classifierPath)
    : detector(std::make_unique<SudokuDetector>(classifierPath))
    {}

    void loop(const std::vector<std::pair<std::filesystem::path, Quad>>& groundTruth) const
    {
        int sudokuNum = 0;
        for (const auto& item : groundTruth) {
            std::cout << "Sudoku " << sudokuNum << std::endl;

            processSudoku(item.first, item.second);
            sudokuNum++;
            if (sudokuNum > 3)
                exit(0);
        }
    }

  private:
    void processSudoku(const fs::path& path, const Quad& gt_bbox) const
    {
        cv::Mat sudokuImg = cv::imread(path.string(), cv::IMREAD_COLOR);
        auto detection = detector->detect(sudokuImg);

        cv::Mat canvas = sudokuImg.clone();

        detection->drawOverlay(canvas);

        auto [scale, resizedCanvas] = resizeMaxSideLen(canvas, 1024);
        cv::imshow("Sudoku", resizedCanvas);
        cv::waitKey();
    }
};

int
main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Missing path to sources. Call as app.exe <path-to-sources>" << std::endl;
        exit(1);
    }

    fs::path root(argv[1]);

    fs::path gtPath = root / "data/sudokus/ground_truth_new.csv";
    if (!fs::exists(gtPath) || !fs::is_regular_file(gtPath)) {
        std::cerr << "The specified gt path is not a file." << std::endl;
        exit(1);
    }

    fs::path classifierPath = root / "share/digit_classifier_ts.onnx";

    auto gt = readGroundTruth(root, gtPath);

    std::cout << "OpenCV Version: " << cv::getVersionString() << std::endl;
    std::cout << "Found " << gt.size() << " ground truth entries" << std::endl;

    const auto app = SudokuApplication(classifierPath.string());
    app.loop(gt);
}
