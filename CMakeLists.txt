cmake_minimum_required(VERSION 3.1)

project(Neon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("External/glfw")

set(source_list
src/main.cpp

include/Neon/NeonCommon.h
src/Neon/NeonCommon.cpp
include/Neon/Neon.h
src/Neon/Neon.cpp
include/Neon/NeonWindow.h
src/Neon/NeonWindow.cpp
include/Neon/NeonShader.h
src/Neon/NeonShader.cpp
include/Neon/NeonImage.h
src/Neon/NeonImage.cpp
include/Neon/NeonTexture.h
src/Neon/NeonTexture.cpp
include/Neon/NeonFrameBufferObject.h
src/Neon/NeonFrameBufferObject.cpp
include/Neon/NeonVertexArrayObject.h
src/Neon/NeonVertexArrayObject.cpp
include/Neon/NeonVertexBufferObject.hpp
include/Neon/NeonRenderData.h
src/Neon/NeonRenderData.cpp
)

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

${source_list})

target_include_directories(Neon PUBLIC 
"External"
"External/glad"
"External/imgui"
"include"
"src"
)

target_link_libraries(Neon glfw)

function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

assign_source_group(${source_list})