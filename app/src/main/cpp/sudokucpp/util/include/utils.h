#ifndef UTILS_H
#define UTILS_H

#include <filesystem>
#include <functional>
#include <geometry.h>
#include <iostream>
#include <iterator>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <vector>

template<typename T>
std::vector<size_t>
argsort(const std::vector<T>& v, std::function<bool(const T&, const T&)> comp)
{

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings when v contains elements of equal values
    std::stable_sort(idx.begin(), idx.end(), [&v, &comp](size_t i1, size_t i2) { return comp(v[i1], v[i2]); });

    return idx;
}

template<typename RandAccessIter, typename T>
std::vector<size_t>
argsort(RandAccessIter begin, RandAccessIter end, std::function<bool(const T&, const T&)> comp)
{
    // initialize original index locations
    std::vector<size_t> idx(std::distance(begin, end));
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings when v contains elements of equal values
    std::stable_sort(idx.begin(), idx.end(), [&begin, &end, &comp](size_t i1, size_t i2) {
        return comp(*(begin + i1), *(begin + i2));
    });

    return idx;
}

struct SudokuGroundTruth
{
    std::filesystem::path imagePath;
    Quad bounds;

    inline SudokuGroundTruth(std::filesystem::path imagePath, Quad bounds)
      : imagePath(std::move(imagePath))
      , bounds(bounds)
    {}
};

std::vector<SudokuGroundTruth>
readGroundTruth(const std::filesystem::path& root, const std::filesystem::path& file);

#endif // UTILS_H
