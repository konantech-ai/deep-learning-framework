set(PROJECT_NAME konanai_engine)

################################################################################
# Source groups
################################################################################
set(include
    "include/vapi.h"
    "include/verrors.h"
    "include/vtypes.h"
    "include/vvalue.h"
    "include/vwrapper.h"
    "include/test.cpp"
)
source_group("include" FILES ${include})

set(api
    "api/vapi.cpp"
    "api/vdump.cpp"
    "api/vcommon.h"
    "api/vdefine.h"
    "api/vconst.h"
    "api/vconst.cpp"
)
source_group("api" FILES ${api})

set(api_objects
    "api_objects/vsession.h"
    "api_objects/vsession.cpp"
    "api_objects/vsession_core.h"
    "api_objects/vmodule.h"
    "api_objects/vmodule.cpp"
    "api_objects/vmodule_core.h"
    "api_objects/vloss.h"
    "api_objects/vloss.cpp"
    "api_objects/vloss_core.h"
    "api_objects/voptimizer.h"
    "api_objects/voptimizer.cpp"
    "api_objects/voptimizer_core.h"
    "api_objects/vtensor.h"
    "api_objects/vtensor.cpp"
    "api_objects/vtensor_core.h"
    "api_objects/vparameters.h"
    "api_objects/vparameters.cpp"
    "api_objects/vparameters_core.h"
    "api_objects/vfunction.cpp"
    "api_objects/vfunction.h"
    "api_objects/vfunction_core.h"
    "api_objects/vmetric.h"
    "api_objects/vmetric.cpp"
    "api_objects/vmetric_core.h"
)
source_group("api_objects" FILES ${api_objects})

set(local_objects
    "local_objects/vgraph.h"
    "local_objects/vgraph.cpp"
    "local_objects/vgraph_core.h"
    "local_objects/vgraph_node.h"
    "local_objects/vgraph_node.cpp"
    "local_objects/vgraph_node_core.h"
    "local_objects/vtensordata.cpp"
    "local_objects/vtensordata.h"
    "local_objects/vtensordata_core.h"
    "local_objects/vexectracer.h"
    "local_objects/vexectracer.cpp"
    "local_objects/vexectracer_core.h"
    "local_objects/vdevicemanager.h"
    "local_objects/vdevicemanager.cpp"
    "local_objects/vdevicemanager_core.h"
    "local_objects/vcbitem.h"
    "local_objects/vcbitem.cpp"
    "local_objects/vcbitem_core.h"
    "local_objects/vcbbackinfo.h"
    "local_objects/vcbbackinfo.cpp"
    "local_objects/vcbbackinfo_core.h"
    "local_objects/vcbbackslot.h"
    "local_objects/vcbbackslot.cpp"
    "local_objects/vcbbackslot_core.h"
    "local_objects/vudfitem.h"
    "local_objects/vudfitem.cpp"
    "local_objects/vudfitem_core.h"
    "local_objects/vexectracerpool.h"
    "local_objects/vexectracerpool.cpp"
    "local_objects/vexectracerpool_core.h"
    "local_objects/vhypermanager.h"
    "local_objects/vhypermanager.cpp"
    "local_objects/vhypermanager_core.h"
)
source_group("local_objects" FILES ${local_objects})

set(support
    "support/vmath.cu" 
    "support/vmath.h"
    "support/vback_queue.h"
    "support/vback_queue.cpp"
    "support/vfunctimer.h"
    "support/vfunctimer.cpp"
)
source_group("support" FILES ${support})

set(utils
    "utils/vexception.cpp"
    "utils/vexception.h"
    "utils/vutils.cpp" 
    "utils/vutils.h"
    "utils/vvalue.cpp"
)
source_group("utils" FILES ${utils})

set(ALL_FILES
    ${include}
    ${api}
    ${api_objects}
    ${local_objects}
    ${support}
    ${utils}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} SHARED ${ALL_FILES} )

set(ROOT_NAMESPACE konanai_engine)

################################################################################
# required package
################################################################################
if(MSVC)
    #opencv should be in install folder!
    find_package(CUDAToolkit) # for library directory information 
else(UNIX)
    #find_package(CUDAToolkit) #CMAKE >= 18
    find_package(OpenCV REQUIRED)
endif()

################################################################################
# Include directories
################################################################################
if(MSVC)
    list(APPEND INCLUDE_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/opencv/build/include"
    )
else(UNIX)
    list(APPEND INCLUDE_DIRECTORY
        ${OpenCV_INCLUDE_DIRS}    
    )
endif()
list(APPEND INCLUDE_DIRECTORY  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" )
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
    "WIN64" "KA_EXPORTS" "KA_WINDOWS" "_WINDLL" "_MBCS" "_CRT_SECURE_NO_WARNINGS" "KA_ENGINE" "V_DEBUG_OBJ_LEAK_OFF"
    )
elseif(UNIX)
    list(APPEND COMPILE_DEF
    "FOR_LINUX"
    )
endif()

if(NO_RANDOM)
    list(APPEND COMPILE_DEF NORANDOM)
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
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<CONFIG:Debug>:
                -g 

                -ftz=false
                -prec-div=true
                -prec-sqrt=true
            >
            -fmad=false
            -cudart static            
       >
   )
elseif(UNIX)
    list(APPEND COMPILE_OPT
    $<$<COMPILE_LANGUAGE:CXX>:
        -ffp-contract=off
        -fpermissive
     >
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:
        -g 
       
        -ftz=false
        -prec-div=true
        -prec-sqrt=true
        >
        -fmad=false
        -Xcompiler "-ffp-contract=off"
    >
    )
endif()
target_compile_options(${PROJECT_NAME} PRIVATE
        ${COMPILE_OPT}
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        ${DEFAULT_CXX_EXCEPTION_HANDLING};
)
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
# Library depency
################################################################################
if(MSVC)
    list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES
        "$<$<CONFIG:Debug>:"
            "opencv_world452d"
        ">"
        "$<$<CONFIG:Release>:"
            "opencv_world452"
        ">"
        cudart_static  curand  "cudnn_ops_infer"
    )
else(UNIX)
    list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES "curand" "stdc++fs" ${OpenCV_LIBS} )
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

################################################################################
# Library path
################################################################################
 if (MSVC)
     list(APPEND LIBRARY_PATH
        "${CMAKE_SOURCE_DIR}/install/opencv/build/${CMAKE_VS_PLATFORM_NAME}/vc14/lib"   
    )
endif()
target_link_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_PATH} ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
