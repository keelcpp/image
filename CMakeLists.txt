cmake_minimum_required(VERSION 3.20)
project(keel_image_formats VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_IMAGE_FORMATS_NO_EXCEPTIONS_RTTI "Compile keel_image_formats without C++ exceptions and RTTI" OFF)
option(KEEL_IMAGE_FORMATS_BUILD_TESTS "Build keel_image_formats unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})
option(KEEL_IMAGE_FORMATS_UNITY_BUILD "Enable unity build on keel_image_formats" OFF)

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_IMAGE_FORMATS_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        ed579b085de734190d16b4054cf7f2d298fd7f6a
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::compress)
    find_package(keel_compress QUIET CONFIG)
    if(TARGET keel::keel_compress AND NOT TARGET keel::compress)
        add_library(keel::compress ALIAS keel::keel_compress)
    endif()
endif()
if(NOT TARGET keel::compress)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../compress/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_COMPRESS)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_COMPRESS "${CMAKE_CURRENT_SOURCE_DIR}/../compress" CACHE PATH "Local keel_compress checkout")
    endif()
    if(KEEL_IMAGE_FORMATS_NO_EXCEPTIONS_RTTI)
        set(KEEL_COMPRESS_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_compress
        GIT_REPOSITORY https://github.com/keelcpp/compress.git
        GIT_TAG        94a9fdd8c554d1a7a735b13c8dd7f0a8daea8628
    )
    FetchContent_MakeAvailable(keel_compress)
endif()

add_library(keel_image_formats STATIC
    src/image.cpp
    src/png.cpp
    src/jpeg.cpp
    src/tiff.cpp
    src/bmp.cpp
    src/tga.cpp
    src/pnm.cpp
    src/gif.cpp
)
add_library(keel::image_formats ALIAS keel_image_formats)

target_link_libraries(keel_image_formats
    PUBLIC  keel::keel
    PRIVATE keel::compress
)
target_compile_features(keel_image_formats PUBLIC cxx_std_20)
set_target_properties(keel_image_formats PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(KEEL_IMAGE_FORMATS_UNITY_BUILD)
    set_target_properties(keel_image_formats PROPERTIES UNITY_BUILD ON)
endif()

target_include_directories(keel_image_formats PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_image_formats PRIVATE /W4 /utf-8)
    if(KEEL_IMAGE_FORMATS_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_image_formats PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_image_formats PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_image_formats PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_IMAGE_FORMATS_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_image_formats PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_image_formats EXPORT keel_image_formatsTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_image_formatsTargets
    FILE keel_image_formatsTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_image_formats
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_image_formatsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_image_formatsConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_image_formats
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_image_formatsConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_image_formatsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_image_formatsConfigVersion.cmake"
    DESTINATION lib/cmake/keel_image_formats
)

if(KEEL_IMAGE_FORMATS_BUILD_TESTS)
    enable_testing()

    add_executable(test_image_formats tests/test_image_formats.cpp)
    target_link_libraries(test_image_formats PRIVATE keel::image_formats)
    target_compile_definitions(test_image_formats PRIVATE
        KEEL_IMAGE_FORMATS_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    )
    if(MSVC)
        target_compile_options(test_image_formats PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_image_formats PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME test_image_formats COMMAND test_image_formats)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
