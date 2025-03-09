# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(TicTacToe)

# Find Torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(tictactoe main.cpp game.h agent.h)

target_include_directories(tictactoe PRIVATE .)
# Link against Torch libraries
target_link_libraries(tictactoe "${TORCH_LIBRARIES}")

# Set C++ standard to 14 or higher (required for PyTorch)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)