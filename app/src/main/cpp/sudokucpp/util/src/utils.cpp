

#include <utils.h>

#include <algorithm>
#include <fstream>
#include <ostream>
#include <sstream>

std::vector<SudokuGroundTruth>
readGroundTruth(const std::filesystem::path& root, const std::filesystem::path& file)
{
    std::ifstream inStream(file);

    if (!inStream) {
        throw std::exception();
    }

    std::vector<SudokuGroundTruth> output;

    std::string line;
    while (std::getline(inStream, line)) {
        std::istringstream lineStream(line);

        std::string p;
        Quad_<double> q;

        std::getline(lineStream, p, ',');
        lineStream >> q;
        std::filesystem::path path = root / p;
        output.emplace_back(path, q);
    }
    return output;
}