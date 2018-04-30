cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# Find the Caffe2 package.
# Caffe2 exports the required targets, so find_package should work for
# the standard Caffe2 installation. If you encounter problems with finding
# the Caffe2 package, make sure you have run `make install` when installing
# Caffe2 (`make install` populates your share/cmake/Caffe2).
find_package(Caffe2 REQUIRED)

if (${CAFFE2_VERSION} VERSION_LESS 0.8.2)
  # Pre-0.8.2 caffe2 does not have proper interface libraries set up, so we
  # will rely on the old path.
  message(WARNING
      "You are using an older version of Caffe2 (version " ${CAFFE2_VERSION}
      "). Please consider moving to a newer version.")
  include(cmake/legacy/legacymake.cmake)
  return()
endif()

# Add compiler flags.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fPIC -Wno-narrowing")

# Print configuration summary.
include(cmake/Summary.cmake)
translation_decoder_print_config_summary()

# add_library(translation_decoder "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")
add_executable(translation_decoder "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")
target_link_libraries(translation_decoder PUBLIC caffe2)
target_include_directories(translation_decoder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
install(TARGETS translation_decoder DESTINATION lib)
