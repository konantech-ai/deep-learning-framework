#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>

#ifndef FOR_LINUX
#include <direct.h>
#endif

#include <Python.h>
#include <eco_api.h>

////////////////////////////////////////////////////////////////
// Exception Handler
////////////////////////////////////////////////////////////////
PyObject* EcoExceptionHandler(char* file_name, int file_line, char* function_name, konan::eco::EcoException ex);
void      ExceptionHandler(char* file_name, int file_line, char* function_name);
