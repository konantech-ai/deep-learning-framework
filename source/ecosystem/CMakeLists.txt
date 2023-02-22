set(PROJECT_NAME konanai_ecosystem)
set(ROOT_NAMESPACE konanai_ecosystem)

################################################################################
# Source groups
################################################################################
set(include
    "include/eco_api.h"
)
source_group("include" FILES ${include})


set(connect
     "connect/eco_api.cpp"
     "connect/tp_nn_server.h"
     "connect/tp_nn_server.cpp"
     "connect/tp_api_conn.h"
     "connect/tp_api_conn.cpp"
     "connect/tp_http_client_sender.h"
     "connect/tp_http_client_sender.cpp"
     "connect/tp_http_client_receiver.h"
     "connect/tp_http_client_receiver.cpp"
)
source_group("connect" FILES ${connect})

set(objects
     "objects/tp_audio_file_reader.h"
     "objects/tp_audio_file_reader.cpp"
     "objects/tp_audio_file_reader_core.h"
     "objects/tp_function.cpp"
     "objects/tp_function.h"
     "objects/tp_function_core.h"
     "objects/tp_loss.cpp"
     "objects/tp_loss.h"
     "objects/tp_loss_core.h"
     "objects/tp_module.h"
     "objects/tp_module.cpp"
     "objects/tp_nn.cpp"
     "objects/tp_nn.h"
     "objects/tp_optimizer.cpp"
     "objects/tp_optimizer.h"
     "objects/tp_optimizer_core.h"
     "objects/tp_param.cpp"
     "objects/tp_param.h"
     "objects/tp_parameters.cpp"
     "objects/tp_parameters.h"
     "objects/tp_scalar.cpp"
     "objects/tp_scalar.h"
     "objects/tp_tensor.cpp"
     "objects/tp_tensor.h"
     "objects/tp_tensor_core.h"
     "objects/tp_parameters_core.h"
     "objects/tp_scalar_core.h"
     "objects/tp_tensordata.cpp"
     "objects/tp_tensordata.h"
     "objects/tp_tensordata_core.h"
     "objects/tp_metric.h"
     "objects/tp_metric.cpp"
     "objects/tp_metric_core.h"
)
source_group("objects" FILES ${objects})

set(utils
     "utils/tp_common.h"
     "utils/tp_values.cpp"
     "utils/tp_utils.cpp"
     "utils/tp_utils.h"
     "utils/tp_math.cpp"
     "utils/tp_math.h"
     "utils/tp_json_parser.h"
     "utils/tp_json_parser.cpp"
     "utils/tp_functimer.h"
     "utils/tp_functimer.cpp"
     "utils/tp_stream.h"
     "utils/tp_stream.cpp"
     "utils/tp_grid_canvas.h"
     "utils/tp_grid_canvas.cpp"
     "utils/tp_cuda.h"
     "utils/tp_cuda.cu"
     "utils/tp_exception.h"
     "utils/tp_exception.cpp"
     "utils/tp_eco_exception_info.h"
     "utils/tp_eco_exception_info.cpp"
)
source_group("utils" FILES ${utils})

set(ALL_FILES
    ${include}
    ${connect}
    ${objects}
    ${utils}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} SHARED ${ALL_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
)

################################################################################
# required package
################################################################################
if(MSVC)
    #opencv should be in install folder!
    find_package(cpprestsdk REQUIRED) 
    find_package(CUDAToolkit) # for library directory information 
else(UNIX)
    #find_package(CUDAToolkit) #CMAKE >= 18
    find_package(cpprestsdk REQUIRED)
    find_package(OpenCV REQUIRED)
endif()

################################################################################
# Include directories
################################################################################
#if(MSVC)
#    list(APPEND INCLUDE_DIRECTORY
#        "C:/ProgramData/Miniconda3/envs/AI-DF/include"
#    )
#else(UNIX)
#    # TODO
#endif()

list(APPEND INCLUDE_DIRECTORY
    "engine"
    "$ENV{KONANAI_PYTHON_PATH}/include"
    "C:/ProgramData/Miniconda3/envs/AI-DF/include"
)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORY})

################################################################################
# Compile definitions
################################################################################
if(MSVC)
   #target_compile_definitions(${PROJECT_NAME} PRIVATE
   list(APPEND COMPILE_DEF 
    "$<$<CONFIG:Debug>:"
        "_DEBUG" 
    ">"
    "WIN64;"  "KA_WINDOWS;" "_CONSOLE;"  "_MBCS" "_CRT_SECURE_NO_WARNINGS" "V_DEBUG_OBJ_LEAK_OFF"
    )
elseif(UNIX)
    list(APPEND COMPILE_DEF
    "FOR_LINUX"
    )
endif()


target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILE_DEF})

################################################################################
# Compile options
################################################################################
if(MSVC)
    list(APPEND COMPILE_OPT
        $<$<COMPILE_LANGUAGE:CXX>:
            /permissive-  /sdl /GS /Zc:wchar_t /Zc:forScope /Zc:inline /Zc:strictStrings- /fp:precise
        > 

   )
elseif(UNIX)
    list(APPEND COMPILE_OPT
    $<$<COMPILE_LANGUAGE:CXX>:
        -ffp-contract=off
        -m64
        -fpermissive
    >

    )
endif()
target_compile_options(${PROJECT_NAME} PRIVATE
        ${COMPILE_OPT}
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        ${DEFAULT_CXX_EXCEPTION_HANDLING};
)

################################################################################
# Library path
################################################################################
if(MSVC)
    list(APPEND LIBRARY_PATH
    # TODO
    )
else(UNIX)
    # TODO
endif()

list(APPEND LIBRARY_PATH
    "${OUTPUT_DIRECTORY}"
)

target_link_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_PATH})

################################################################################
# LInker options
################################################################################
if(MSVC)
    list(APPEND ${LINKER_OPT}
        $<$<CONFIG:Debug>:
        >
        $<$<CONFIG:Release>:
            /OPT:REF; /OPT:ICF
        >
        /DYNAMICBASE /NXCOMPAT
    )
endif()
 target_link_options(${PROJECT_NAME} PRIVATE ${LINKER_OPT})


################################################################################
# Link Dependencies
################################################################################
if(MSVC)
    list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES
        "$<$<CONFIG:Debug>:"
            "opencv_world452d"
        ">"
        "$<$<CONFIG:Release>:"
            "opencv_world452"
        ">"
        "cpprestsdk::cpprest"
        "konanai_engine"
        cudart_static  curand  "cudnn_ops_infer"
    )
else(UNIX)
    list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES "curand" "stdc++fs" "crypto" "konanai_engine" "cpprestsdk::cpprest" ${OpenCV_LIBS} )
endif()

add_dependencies(${PROJECT_NAME}
    konanai_engine
)

target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
