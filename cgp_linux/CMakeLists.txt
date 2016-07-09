cmake_minimum_required(VERSION 2.8)
project(CGP_action_1)

add_subdirectory(src)
include_directories(src)

aux_source_directory(. DIR_SRCS)
add_executable(CGP_action_1 ${DIR_SRCS})

find_package(OpenCV REQUIRED)
set(OPENCV_INCLUDES_DIR /home/szy/opencv_310/include)
set(OPENCV_LIBS_DIR /home/szy/opencv_310/lib)

include_directories(${OPENCV_INCLUDES_DIR})
#link_directories(${OPENCV_LIBS_DIR})

target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_highgui.so)
target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_imgcodecs.so)
target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_core.so)
target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_imgproc.so)
target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_optflow.so)
target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_video.so)
target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_xfeatures2d.so)
target_link_libraries(CGP_action_1 ${OPENCV_LIBS_DIR}/libopencv_features2d.so)
target_link_libraries(CGP_action_1 cgp_mat)
target_link_libraries(CGP_action_1 node_function)
