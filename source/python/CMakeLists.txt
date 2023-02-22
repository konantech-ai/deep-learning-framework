set(PROJECT_NAME konanai_python)
set(ROOT_NAMESPACE konanai_python)

################################################################################
# Source groups
################################################################################

set(cpp
    "../ecosystem/utils/tp_values.cpp"
    "cpp/api_python.cpp"
    "cpp/api_python.h"
    "cpp/data_handler.cpp"
    "cpp/data_handler.h"
    "cpp/exception_handler.cpp"
    "cpp/exception_handler.h"
    "cpp/inner_proc.cpp"
    "cpp/inner_proc.h"
)
source_group("cpp" FILES ${cpp})

set(ALL_FILES
    ${cpp}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} SHARED ${ALL_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
)

################################################################################
# Required package
###############################################################################
if(MSVC)
    find_package(cpprestsdk REQUIRED)
else(UNIX)
    find_package(cpprestsdk REQUIRED)
endif()

################################################################################
# Include directories
################################################################################
if(MSVC)
    list(APPEND ADDITIONAL_INCLUDE_DIRECTORIES
        "$ENV{KONANAI_PYTHON_PATH}/include"
    )
else(UNIX)
    list(APPEND ADDITIONAL_INCLUDE_DIRECTORIES
        "/usr/include/python3.8/"
    )
endif()

list(APPEND ADDITIONAL_INCLUDE_DIRECTORIES
    "../engine"
    "../ecosystem/include"
)

target_include_directories(${PROJECT_NAME} PUBLIC ${ADDITIONAL_INCLUDE_DIRECTORIES})

################################################################################
# Library paths
################################################################################
if(MSVC)
    list(APPEND ADDITIONAL_LIBRARY_DIRECTORIES
        "$ENV{KONANAI_PYTHON_PATH}/libs"
    )
else(UNIX)
    # TODO
endif()

list(APPEND ADDITIONAL_LIBRARY_DIRECTORIES
    "${OUTPUT_DIRECTORY}"
)

target_link_directories(${PROJECT_NAME} PUBLIC ${ADDITIONAL_LIBRARY_DIRECTORIES})

################################################################################
# Link dependencies
################################################################################
if(MSVC)
    list(APPEND ADDITIONAL_LIBS
#        "$<$<CONFIG:Debug>:"
#            "opencv_world452d"
#        ">"
#        "$<$<CONFIG:Release>:"
#            "opencv_world452"
#        ">"
        "cpprestsdk::cpprest"
        "python39"
    )
else(UNIX)
    list(APPEND ADDITIONAL_LIBS
        "cpprestsdk::cpprest"
         )
endif()

list(APPEND ADDITIONAL_LIBS
    "konanai_engine"
    "konanai_ecosystem"
)

target_link_libraries(${PROJECT_NAME} PUBLIC "${ADDITIONAL_LIBS}")

add_dependencies(${PROJECT_NAME}
    konanai_engine
    konanai_ecosystem
)

################################################################################
# Compile definitions
################################################################################
if(MSVC)
   #target_compile_definitions(${PROJECT_NAME} PRIVATE
   list(APPEND COMPILE_DEF 
    "$<$<CONFIG:Debug>:"
        "_DEBUG" 
    ">"
    "$<$<CONFIG:Release>:"
        "NDEBUG"
    ">"
    "WIN64" "KA_WINDOWS" "KPY_EXPORTS" "_CONSOLE" "_MBCS" "_CRT_SECURE_NO_WARNINGS"
    )
elseif(UNIX)
    list(APPEND COMPILE_DEF
    "FOR_LINUX"
    "KPY_EXPORTS"
    )
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILE_DEF})

################################################################################
# Compile options
################################################################################
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            $<$<CONFIG:Debug>:  
                /Od /MDd /Zi
            >
            $<$<CONFIG:Release>:
                /O2 /Oi /Gy /MD /Zi
            >
            /permissive- /EHsc /W3 /sdl /GS /Zc:wchar_t /Zc:forScope /Zc:inline /Zc:strictStrings- /fp:precise
        > 
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<CONFIG:Debug>:  
                -g 
                -G 
                -Xcompiler "/MDd /EHsc /Zi"
            >
            $<$<CONFIG:Release>:
               -Xcompiler "/MD /EHsc"
            >
            --machine 64 
            --compile 
        >
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        ${DEFAULT_CXX_EXCEPTION_HANDLING};
    )
else(UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        ${DEFAULT_CXX_EXCEPTION_HANDLING};
         -fpermissive
    )
endif()

################################################################################
# Linker options
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
# Post build event
################################################################################

if(MSVC)
#    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../python)
#    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy_directory
#                       ${CMAKE_SOURCE_DIR}/source/python/konanai    ${CMAKE_BINARY_DIR}/../python/konanai)
#
#    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/source/python/requirements.txt 
#                                                     ${CMAKE_SOURCE_DIR}/source/python/setup.py 
#                                                     ${CMAKE_BINARY_DIR}/../python)
#
#    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/../konanai_python.dll
#                                                     ${CMAKE_BINARY_DIR}/../cpprest_2_10.dll
#                                                     ${CMAKE_BINARY_DIR}/../konanai_engine.dll
#                                                     ${CMAKE_BINARY_DIR}/../libcrypto-1_1-x64.dll
#                                                     ${CMAKE_BINARY_DIR}/../libssl-1_1-x64.dll
#                                                     ${CMAKE_BINARY_DIR}/../opencv_world452.dll
#                                                     ${CMAKE_BINARY_DIR}/../opencv_world452d.dll
#                                                     ${CMAKE_BINARY_DIR}/../python39.dll
#                                                     ${CMAKE_BINARY_DIR}/../python/konanai)
else(UNIX)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../python)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/source/python/konanai    ${CMAKE_BINARY_DIR}/../python/konanai)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/source/python/requirements.txt 
                                                     ${CMAKE_SOURCE_DIR}/source/python/setup.py 
                                                     ${CMAKE_BINARY_DIR}/../python)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/../libkonanai_python.so  
                                                     ${CMAKE_BINARY_DIR}/../python/konanai)
endif()