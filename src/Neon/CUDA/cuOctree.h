#pragma once

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <Neon/Neon.h>

void RunOctreeExample(Neon::Scene* scene);