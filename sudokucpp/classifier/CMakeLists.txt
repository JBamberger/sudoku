cmake_minimum_required(VERSION 3.14)

project(digit_classifier)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED core imgcodecs imgproc)

#find_package(Torch REQUIRED)
#find_package(TorchVision REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(digit_classifier
        "test_classifier.cpp"
        CellClassifier.cpp include/CellClassifier.h)
target_include_directories(digit_classifier PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        #        ${TORCH_INCLUDE_DIRS}
        include
        )
target_link_libraries(digit_classifier PRIVATE
        #        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        #        TorchVision::TorchVision
        )

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
#if (MSVC)
#    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#    get_target_property(TV_DLL TorchVision::TorchVision IMPORTED_LOCATION_RELEASE)
#    list(APPEND TORCH_DLLS ${TV_DLL})
#    add_custom_command(TARGET digit_classifier
#            POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            ${TORCH_DLLS}
#            $<TARGET_FILE_DIR:digit_classifier>)
#endif (MSVC)
