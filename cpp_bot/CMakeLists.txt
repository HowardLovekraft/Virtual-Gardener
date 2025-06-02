cmake_minimum_required(VERSION 3.10.2)
project(AI_Sadovnik CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS SSL Crypto) 
find_package(Boost REQUIRED COMPONENTS system)
find_package(fmt CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(TgBot REQUIRED)

add_executable(AI_Sadovnik main.cpp)

target_link_libraries(AI_Sadovnik PRIVATE
    fmt::fmt
    ${CMAKE_THREAD_LIBS_INIT}
    TgBot::TgBot
    CURL::libcurl
    Boost::system
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
)