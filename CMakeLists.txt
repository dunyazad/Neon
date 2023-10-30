cmake_minimum_required(VERSION 3.1)

project(Neon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("External/glfw")
add_subdirectory("External/glm")
#add_subdirectory("External/OpenMesh")
add_subdirectory("External/tinyply")

set(source_list
src/main.cpp

include/Neon/Neon.h
src/Neon/Neon.cpp
include/Neon/NeonCommon.h
src/Neon/NeonCommon.cpp
include/Neon/NeonDebugEntity.h
src/Neon/NeonDebugEntity.cpp
include/Neon/NeonEntity.h
src/Neon/NeonEntity.cpp
include/Neon/NeonEvent.h
src/Neon/NeonEvent.cpp
include/Neon/NeonFrameBufferObject.h
src/Neon/NeonFrameBufferObject.cpp
include/Neon/NeonImage.h
src/Neon/NeonImage.cpp
include/Neon/NeonScene.h
src/Neon/NeonScene.cpp
include/Neon/NeonTriangulator.h
src/Neon/NeonTriangulator.cpp
include/Neon/NeonURL.h
src/Neon/NeonURL.cpp
include/Neon/NeonVertexArrayObject.h
src/Neon/NeonVertexArrayObject.cpp
include/Neon/NeonVertexBufferObject.hpp
include/Neon/NeonWindow.h
src/Neon/NeonWindow.cpp
include/Neon/TriangleTriangleOverlap.h
src/Neon/TriangleTriangleOverlap.cpp

include/Neon/Component/NeonCamera.h
src/Neon/Component/NeonCamera.cpp
include/Neon/Component/NeonCameraManipulator.h
src/Neon/Component/NeonCameraManipulator.cpp
include/Neon/Component/NeonComponent.h
src/Neon/Component/NeonComponent.cpp
include/Neon/Component/NeonLight.h
src/Neon/Component/NeonLight.cpp
include/Neon/Component/NeonMesh.h
src/Neon/Component/NeonMesh.cpp
include/Neon/Component/NeonShader.h
src/Neon/Component/NeonShader.cpp
include/Neon/Component/NeonTexture.h
src/Neon/Component/NeonTexture.cpp
include/Neon/Component/NeonTransform.h
src/Neon/Component/NeonTransform.cpp
include/Neon/Component/NeonVETM.h
src/Neon/Component/NeonVETM.cpp

include/Neon/Component/SpatialPartitioning/NeonBSPTree.hpp
include/Neon/Component/SpatialPartitioning/NeonRegularGrid.h
src/Neon/Component/SpatialPartitioning/NeonRegularGrid.cpp
include/Neon/Component/SpatialPartitioning/NeonSpatialHashing.h
src/Neon/Component/SpatialPartitioning/NeonSpatialHashing.cpp

include/Neon/System/NeonEntityUpdateSystem.h
src/Neon/System/NeonEntityUpdateSystem.cpp
include/Neon/System/NeonEventSystem.h
src/Neon/System/NeonEventSystem.cpp
include/Neon/System/NeonRenderSystem.h
src/Neon/System/NeonRenderSystem.cpp
include/Neon/System/NeonSystem.h
src/Neon/System/NeonSystem.cpp
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
"External/earcut.hpp/include"
"External/glad"
"External/glm"
"External/imgui"
"External/json/include"
"include"
"src"
)

target_link_libraries(Neon glfw)
#target_link_libraries(Neon $<$<CONFIG:Debug>:OpenMeshCored>)
#target_link_libraries(Neon $<$<CONFIG:Release>:OpenMeshCore>)

target_link_libraries(Neon tinyply)

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