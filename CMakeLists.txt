cmake_minimum_required(VERSION 3.1)

project(Neon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("External/glfw")

#add_library(Neon
add_executable(Neon
External/glad/glad/gl.c

External/imgui/imgui.cpp
External/imgui/imgui_demo.cpp
External/imgui/imgui_draw.cpp
External/imgui/imgui_tables.cpp
External/imgui/imgui_widgets.cpp
External/imgui/backends/imgui_impl_opengl3.cpp
External/imgui/backends/imgui_impl_glfw.cpp

src/NeonCommon.h
src/NeonCommon.cpp
src/Neon.h
src/Neon.cpp
src/NeonWindow.h
src/NeonWindow.cpp
src/NeonShader.h
src/NeonShader.cpp
src/main.cpp)

target_include_directories(Neon PUBLIC "External/glad")
target_include_directories(Neon PUBLIC "External/imgui")

target_link_libraries(Neon glfw)
