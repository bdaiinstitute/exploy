# This will define the following variables:
#   onnxruntime_FOUND        -- True if the system has the onnxruntime library
#   onnxruntime_INCLUDE_DIRS -- The include directories for onnxruntime
#   onnxruntime_LIBRARIES    -- Libraries to link against
#   onnxruntime_CXX_FLAGS    -- Additional (required) compiler flags

include(FindPackageHandleStandardArgs)


set(onnxruntime_INSTALL_PREFIX $ENV{CONDA_PREFIX})
set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime)
set(onnxruntime_LIBRARIES onnxruntime)
set(onnxruntime_CXX_FLAGS "") # no flags needed

find_library(onnxruntime_LIBRARY
    NAMES onnxruntime
    PATHS "${onnxruntime_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH
)

if(onnxruntime_LIBRARY)
    add_library(onnxruntime SHARED IMPORTED)
    set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
    set_property(TARGET onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIRS}")
    set_property(TARGET onnxruntime PROPERTY INTERFACE_COMPILE_OPTIONS "${onnxruntime_CXX_FLAGS}")
endif()

find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_LIBRARY onnxruntime_INCLUDE_DIRS)
