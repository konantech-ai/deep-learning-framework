#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>

#ifndef FOR_LINUX
#include <direct.h>
#endif

////////////////////////////////////////////////////////////////
// Inner procedure - added by taebin.heo on 2023.01.06
////////////////////////////////////////////////////////////////
PyObject* GetEngineVersion(void* session_ptr);

void      SetNoTracer(void* session_ptr);
void      UnsetNoTracer(void* session_ptr);

PyObject* GetBuiltinNames(void* session_ptr);
bool      IsBuiltinName(void* session_ptr, PyObject* domain_str, PyObject* builtin_str);
PyObject* GetLayerFormula(void* session_ptr, PyObject* layer_name_str);

PyObject* GetLeakInfo(void* session_ptr, bool flag_bool);
void      DumpLeakInfo(void* session_ptr, bool flag_bool);

void*     CreateUserDefinedLayer(void* session_ptr, PyObject* name_str, PyObject* formula_str, PyObject* parameter_dict, PyObject* args_dict);
void*     CreateModel(void* session_ptr, PyObject* model_name_str, PyObject* args_dict);
void      SaveModule(void* session_ptr, void* module_ptr, PyObject* file_name_str); // taebin.heo : question
void*     LoadModule(void* session_ptr, PyObject* file_name_str);
