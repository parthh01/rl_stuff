# build.sh
export CMAKE_PREFIX_PATH=/usr/local/Cellar/pytorch/2.5.1_4
cmake -S . -B build
cmake --build build
