cmake_minimum_required(VERSION 3.16)
project(distgpu_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4 /utf-8)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# IXWebSocket: до FetchContent, чтобы подпроект подхватил CACHE
set(IXWEBSOCKET_USE_TLS OFF CACHE BOOL "Build IXWebSocket with TLS" FORCE)
set(IXWEBSOCKET_USE_ZLIB OFF CACHE BOOL "Build IXWebSocket with zlib" FORCE)
set(IXWEBSOCKET_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(IXWEBSOCKET_BUILD_TEST OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  ixwebsocket
  URL https://github.com/machinezone/IXWebSocket/archive/refs/tags/v11.4.6.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(ixwebsocket json)

add_executable(distgpu-worker src/main.cpp)

target_include_directories(distgpu-worker PRIVATE
  "${json_SOURCE_DIR}/single_include"
)

target_link_libraries(distgpu-worker PRIVATE ixwebsocket)

if(WIN32)
  target_link_libraries(distgpu-worker PRIVATE ws2_32 crypt32)
else()
  target_link_libraries(distgpu-worker PRIVATE pthread)
endif()

install(TARGETS distgpu-worker RUNTIME DESTINATION bin)
