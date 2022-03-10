

#include <utils.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <ostream>
#include <sstream>
#include <utility>

std::istream&
operator>>(std::istream& str, Quad& data)
{
    char delim;
    Quad tmp;
    if (str >> tmp.corners.at(0).x >> delim >> tmp.corners.at(0).y >> delim //
        >> tmp.corners.at(1).x >> delim >> tmp.corners.at(1).y >> delim     //
        >> tmp.corners.at(2).x >> delim >> tmp.corners.at(2).y >> delim     //
        >> tmp.corners.at(3).x >> delim >> tmp.corners.at(3).y) {

        data = std::move(tmp);
    } else {
        str.setstate(std::ios::failbit);
    }
    return str;
}

std::ostream&
operator<<(std::ostream& str, const Quad& data)
{
    std::string delim = ", ";
    str << data.corners.at(0).x << delim << data.corners.at(0).y << delim //
        << data.corners.at(1).x << delim << data.corners.at(1).y << delim //
        << data.corners.at(2).x << delim << data.corners.at(2).y << delim //
        << data.corners.at(3).x << delim << data.corners.at(3).y;
    return str;
}

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
        Quad q;

        std::getline(lineStream, p, ',');
        lineStream >> q;
        std::filesystem::path path = root / p;
        output.emplace_back(path, q);
    }
    return output;
}