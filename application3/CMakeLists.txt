CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(PROJET)

ADD_EXECUTABLE(application3 application3.cpp ../maxflow/graph.cpp)
TARGET_LINK_LIBRARIES(application3 ${OpenCV_LIBS})
