cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)

project(HOUGH-TRANSFORM VERSION 1.0 LANGUAGES CXX CUDA)

# Encontrar SDL2
find_package(SDL2 REQUIRED)

# Configuración manual de SDL2_image
include_directories(/usr/include/SDL2) # Ruta a los encabezados de SDL2 y SDL2_image
set(SDL2_IMAGE_LIBRARY /usr/lib/x86_64-linux-gnu/libSDL2_image.so) # Ruta a la biblioteca compartida

# Encontrar OpenCV
find_package(OpenCV REQUIRED)

# Incluir directorios de encabezados
include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

# Incluir encabezados de las bibliotecas locales
file(GLOB_RECURSE LIB_HEADERS lib/*.h)
foreach(FILE ${LIB_HEADERS})
    get_filename_component(DIR_PATH ${FILE} PATH)
    include_directories(${DIR_PATH})
endforeach()

# Buscar archivos fuente
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu lib/*.cpp lib/*.cu)

# Crear el ejecutable
add_executable(HOUGH-TRANSFORM ${SRC_FILES})

# Enlazar bibliotecas
target_link_libraries(HOUGH-TRANSFORM ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY} ${OpenCV_LIBS})

# Habilitar compilación separable para CUDA
set_property(TARGET HOUGH-TRANSFORM 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Definir macro para compilación CUDA
target_compile_definitions(HOUGH-TRANSFORM PRIVATE CUDA_COMPILE)

# Copiar los assets al directorio de compilación
file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})
