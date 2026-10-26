cmake_minimum_required(VERSION 3.12)
project(emd_c C)

# Imposta la directory di output per la libreria compilata
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Aggiungi la libreria C
add_library(emd SHARED emd.c)

# Imposta il suffisso corretto per la libreria condivisa
if (WIN32)
    set_target_properties(emd PROPERTIES SUFFIX ".dll")
else()
    set_target_properties(emd PROPERTIES SUFFIX ".so")
endif()

#if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
#    set_target_properties(emd PROPERTIES SUFFIX ".dll")
#elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
#    set_target_properties(emd PROPERTIES SUFFIX ".dylib")
#else()
#    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
#endif()






# Nome del progetto
#cmake_minimum_required(VERSION 3.12)
#project(emd)

# Imposta il linguaggio
#set(CMAKE_C_STANDARD 99)

# Crea una libreria condivisa dal file emd.c
#add_library(emd MODULE emd.c)

# Imposta il nome della libreria condivisa
#set_target_properties(emd PROPERTIES PREFIX "lib" SUFFIX ".so")
#set_target_properties(emd PROPERTIES PREFIX "" OUTPUT_NAME "emd")

# Aggiungi definizioni o flag specifici se necessario (opzionale)
# target_compile_definitions(emd PRIVATE SOME_DEFINITION)
# target_compile_options(emd PRIVATE -Wall)
