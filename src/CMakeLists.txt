set( OpenCV_DIR D:/opencv/opencv/build )
FIND_PACKAGE( OpenCV REQUIRED )

add_subdirectory(basic)

ADD_EXECUTABLE( search_circle main.cpp )
TARGET_LINK_LIBRARIES( search_circle PUBLIC ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES( search_circle PUBLIC basic)
target_include_directories(search_circle PUBLIC
                          "${PROJECT_SOURCE_DIR}/src/basic"
                          )