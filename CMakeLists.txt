cmake_minimum_required(VERSION 3.30)
project(Renderer CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer legacy GL if needed
set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(assimp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswscale libswresample)

add_executable(renderer main.cpp)

target_include_directories(renderer PRIVATE
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${ASSIMP_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(renderer
    OpenGL::GL
    ${GLEW_LIBRARIES}
    glfw
    assimp
    ${FFMPEG_LIBRARIES}
    m pthread dl z lzma
)


