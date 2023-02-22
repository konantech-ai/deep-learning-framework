cmake_minimum_required (VERSION 3.9)
project ("konanai")

# options
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation")
option(NO_RANDOM "Disable random(shuffle, layer initialize)")

# add subdirectory
add_subdirectory(source)
add_subdirectory(docs)

