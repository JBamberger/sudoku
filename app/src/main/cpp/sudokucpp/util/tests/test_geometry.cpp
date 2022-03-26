#include "geometry.h"

#include "gtest/gtest.h"

TEST(geometry, test_quad_constructor_noparams)
{
    Quad quad;
    EXPECT_EQ(quad.corners[0], cv::Point2d(0, 0));
    EXPECT_EQ(quad.corners[1], cv::Point2d(0, 0));
    EXPECT_EQ(quad.corners[2], cv::Point2d(0, 0));
    EXPECT_EQ(quad.corners[3], cv::Point2d(0, 0));
}

TEST(geometry, test_quad_constructor_fourpoints)
{
    Quad quad({ 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 });
    EXPECT_EQ(quad.corners[0], cv::Point2d(1, 2));
    EXPECT_EQ(quad.corners[1], cv::Point2d(3, 4));
    EXPECT_EQ(quad.corners[2], cv::Point2d(5, 6));
    EXPECT_EQ(quad.corners[3], cv::Point2d(7, 8));
}


TEST(geometry, test_area_cw)
{
    Quad quad1({ 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 });
    EXPECT_FLOAT_EQ(quad1.area(), 1.0);

    Quad quad2({ 1, 1 }, { 2, 1 }, { 1, 0 }, { 0, 0 });
    EXPECT_FLOAT_EQ(quad2.area(), 1.0);
}

TEST(geometry, test_area_ccw)
{
    Quad quad({ 0, 1 }, { 0, 0 }, { 1, 0 }, { 1, 1 });
    EXPECT_FLOAT_EQ(quad.area(), 1.0);
}