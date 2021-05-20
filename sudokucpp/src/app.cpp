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
    explicit SudokuApplication()
      : detector(std::make_unique<SudokuDetector>())
    {}

    void loop(const std::vector<std::pair<std::filesystem::path, Quad>>& groundTruth) const
    {
        int sudokuNum = 0;
        for (const auto& item : groundTruth) {
            std::cout << "Sudoku " << sudokuNum << std::endl;

            processSudoku(item.first, item.second);

            sudokuNum++;
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
main()
{
    auto gt = readGroundTruth(sudokusGtPath);

    std::cout << "OpenCV Version: " << cv::getVersionString() << std::endl;
    std::cout << "Found " << gt.size() << " ground truth entries" << std::endl;

    const auto app = SudokuApplication();
    app.loop(gt);
}
