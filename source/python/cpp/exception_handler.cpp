#include "exception_handler.h"

////////////////////////////////////////////////////////////////

PyObject* EcoExceptionHandler(char* file_name, int file_line, char* function_name, konan::eco::EcoException ex) {
	PyObject* ErrorCode;
	PyGILState_STATE gil_state = PyGILState_Ensure();
	ErrorCode = PyLong_FromLongLong((int64)VERR_OK);
	PyGILState_Release(gil_state);

	try {
		// User Exception : m_core is NULL
		if ( ex.get_core() == NULL ) {
			printf("\nC++ Exception( EcoException ) : in %s : %d line : %s() : %s\n", file_name, file_line, function_name, "EcoException m_core is NULL");
			gil_state = PyGILState_Ensure();
			ErrorCode = PyLong_FromLongLong((int64)VERR_INVALID_CORE);
			PyGILState_Release(gil_state);
		}
		// User Exception : Get Error Code
		else {
			//printf("A4\n");
			//printf("\nC++ Exception(ErrorCode : %d) : in %s : %d line : %s() : %s\n", ex.get_error_code(), file_name, file_line, function_name, ex.get_error_message(4).c_str());
			printf("Engine Exception (오류코드: %d)\n", ex.get_error_code());
			//printf("DP2\n");
			//printf("in %s\n", file_name);
			//printf("DP3\n");
			//printf("%d line\n", file_line);
			//printf("DP4\n");
			//printf("%s()\n", function_name);
			//printf("DP5\n");
			printf("%s\n", ex.get_error_message(4).c_str());
			//printf("DP6\n");
			gil_state = PyGILState_Ensure();
			ErrorCode = PyLong_FromLongLong( (int64)(ex.get_error_code()) );
			PyGILState_Release(gil_state);
		}
	}
	// User Exception : Out of EcoException
	catch (...) {
		printf("\nC++ Exception( EcoException ) : in %s : %d line : %s() : %s\n", file_name, file_line, function_name, "Out of EcoException");
		gil_state = PyGILState_Ensure();
		ErrorCode = PyLong_FromLongLong((int64)VERR_UNKNOWN);
		PyGILState_Release(gil_state);
	}

	return ErrorCode;
}

void ExceptionHandler(char* file_name, int file_line, char* function_name) {
	PyObject* ErrorCode;
	PyGILState_STATE gil_state = PyGILState_Ensure();
	ErrorCode = PyLong_FromLongLong((int64)VERR_OK);
	PyGILState_Release(gil_state);

	try {
		throw;
	}
	// User Exception Control
	catch (konan::eco::EcoException ex) {
		ErrorCode = EcoExceptionHandler(file_name, file_line, function_name, ex);
	}
	// Standard Exception Control
	catch (std::exception& e) {
		printf("\nC++ Exception( std::exception ) : in %s : %d line : %s() : %s\n", file_name, file_line, function_name, e.what());
		gil_state = PyGILState_Ensure();
		ErrorCode = PyLong_FromLongLong((int64)VERR_STD_ERROR);
		PyGILState_Release(gil_state);
	}
	// Out of Standard Exception Control
	catch (...) {
		printf("\nC++ Exception( std::exception ) : in %s : %d line : %s() : %s\n", file_name, file_line, function_name, "Out of std::exception");
		gil_state = PyGILState_Ensure();
		ErrorCode = PyLong_FromLongLong((int64)VERR_UNKNOWN);
		PyGILState_Release(gil_state);
	}

	// For Python API
	gil_state = PyGILState_Ensure();
	PyErr_SetObject(PyExc_BaseException, ErrorCode);
	PyGILState_Release(gil_state);

}
