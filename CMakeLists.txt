cmake_minimum_required(VERSION 3.1)

project(BlockA)

add_subdirectory("External/glfw")

add_executable(BlockA
External/glad/glad/gl.c

External/imgui/imgui.cpp
External/imgui/imgui_demo.cpp
External/imgui/imgui_draw.cpp
External/imgui/imgui_tables.cpp
External/imgui/imgui_widgets.cpp
External/imgui/backends/imgui_impl_opengl3.cpp
External/imgui/backends/imgui_impl_glfw.cpp

src/BlockA_Common.h
src/BlockA.h
src/BlockA.cpp
src/main.cpp)

target_include_directories(BlockA PUBLIC "External/glad")
target_include_directories(BlockA PUBLIC "External/imgui")

target_link_libraries(BlockA glfw)
