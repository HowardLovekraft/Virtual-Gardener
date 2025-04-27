cmake_minimum_required(VERSION 3.10.2)
cmake_policy(SET CMP0167 NEW)
project(AI_Sadovnik)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(Boost_USE_MULTITHREADED ON)

set(CMAKE_TOOLCHAIN_FILE "D:/VS code/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(CURL REQUIRED)
set(CMAKE_PREFIX_PATH "D:/VS code/Projects/vcpkg/installed/x64-windows" ${CMAKE_PREFIX_PATH})
find_package(nlohmann_json CONFIG REQUIRED)

include_directories(${OPENSSL_INCLUDE_DIR} ${Boost_INCLUDE_DIR})

if (CURL_FOUND)
    include_directories(${CURL_INCLUDE_DIRS})
    add_definitions(-DHAVE_CURL)
endif()

add_executable(AI_Sadovnik main.cpp)

set(CMAKE_WINDOWS_TARGET_PLATFORM_MIN "11.0")
target_compile_definitions(AI_Sadovnik PRIVATE _WIN32_WINNT=0x0A00)

set(TGBOT_LIB_PATH "D:/VS code/Projects/vcpkg/installed/x64-windows/lib/TgBot.lib")
set(CURL_LIB_PATH "D:/VS code/Projects/vcpkg/installed/x64-windows/lib/libcurl.lib")
set(BOOST_SYSTEM_LIB_PATH "D:/VS code/Projects/vcpkg/installed/x64-windows/lib/boost_system-vc143-mt-x64-1_87.lib")
set(OPENSSL_SSL_LIB_PATH "D:/VS code/Projects/vcpkg/installed/x64-windows/lib/libssl.lib")
set(OPENSSL_CRYPTO_LIB_PATH "D:/VS code/Projects/vcpkg/installed/x64-windows/lib/libcrypto.lib")

target_link_libraries(AI_Sadovnik PRIVATE fmt::fmt ${CMAKE_THREAD_LIBS_INIT}
    "${TGBOT_LIB_PATH}"
    CURL::libcurl
    "${BOOST_SYSTEM_LIB_PATH}"
    "${OPENSSL_SSL_LIB_PATH}"
    "${OPENSSL_CRYPTO_LIB_PATH}"
    nlohmann_json::nlohmann_json
)