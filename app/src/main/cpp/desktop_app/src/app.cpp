#include <SudokuDetector.h>

#include <config.h>
#include <drawutil.h>
#include <utils.h>

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

class SudokuDetectionBenchmark
{
    std::unique_ptr<SudokuDetector> detector;

  public:
    explicit SudokuDetectionBenchmark(const std::string& classifierPath)
      : detector(std::make_unique<SudokuDetector>(classifierPath))
    {}

    void loop(const std::vector<SudokuGroundTruth>& groundTruth) const
    {
        size_t sudokuNum = 0;
        while (true) {

            const auto& gtSudoku = groundTruth[sudokuNum];
            std::cout << "Sudoku " << sudokuNum << " Path: " << gtSudoku.imagePath.string() << std::endl;

            cv::Mat sudokuImg = cv::imread(gtSudoku.imagePath.string(), cv::IMREAD_COLOR);
            auto detection = detector->detect(sudokuImg);

            cv::Mat canvas = sudokuImg.clone();

            if (detection->foundSudoku) {
                detection->drawOverlay(canvas);
            } else{
                std::cerr << "Failed to detect sudoku!" << std::endl;
            }

            auto [scale, resizedCanvas] = resizeMaxSideLen(canvas, 1024);

            cv::imshow("Sudoku", resizedCanvas);
            int key = cv::waitKey();

            switch (key) {
                case -1:
                case 'q':
                    return;
                case 'n':
                    sudokuNum++;
                    break;
                case 'p':
                    sudokuNum--;
                    break;
                default:
                    sudokuNum++;
                    // std::cout << "Pressed key: " << key << std::endl;
            }
            if (sudokuNum >= groundTruth.size()) {
                sudokuNum = groundTruth.size() - 1;
                std::cout << "Reached last Sudoku. Press q to quit." << std::endl;
            } else if (sudokuNum < 0) {
                sudokuNum = 0;
                std::cout << "Reached first Sudoku. Press q to quit." << std::endl;
            }
        }
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

    const auto app = SudokuDetectionBenchmark(classifierPath.string());
    app.loop(gt);
}
