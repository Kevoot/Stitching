project( STITCHING )
cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )


message( STATUS "OpenCV library status:" )
message( STATUS "    version: ${OpenCV_VERSION}" )
message( STATUS "    libraries: ${OpenCV_LIBS}" )
message( STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" )

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( ${PROJECT_NAME} stitching.cpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )