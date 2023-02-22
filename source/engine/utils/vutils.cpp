#include "../utils/vutils.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vmodule.h"
#include "../api/vcommon.h"

//hs.cho
#ifdef FOR_LINUX
#define strcpy_s(a,b,c) !strncpy(a,c,b)
#define strtok_s strtok_r
#endif
VUtils vutils;;

VValue VUtils::seek_dict(VDict dict, string sKey, VValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) return kDefaultValue;
	else return dict[sKey];
}

VValue VUtils::seek_dict_set(VDict dict, string sKey, VValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) {
		dict[sKey] = kDefaultValue;
		return kDefaultValue;
	}
	else return dict[sKey];
}

VValue VUtils::seek_dict_reset(VDict dict, string sKey, VValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) return kDefaultValue;
	VValue value = dict[sKey];
	dict.erase(sKey);
	return value;
}

string VUtils::tolower(string str) {
	//hs.cho
	std::transform(str.begin(), str.end(), str.begin(),  ::tolower);
	return str;
}

VTensorDict VUtils::toTensorDict(VSession session, VDict dict) {
	VTensorDict tensors;
	
	bool bCpuFound = false;
	bool bGpuFound = false;

	for (auto& it : dict) {
		VTensor tensor(session, (VHTensor)(VHandle)it.second);
		tensors[it.first] = tensor;
		if (tensor.device() < 0) bCpuFound = true;
		else bGpuFound = true;
	}

	if (bCpuFound && bGpuFound) {
		for (auto& it : tensors) {
			VTensor tensor = it.second;
			if (tensor.device() < 0) {
				tensor = tensor.toDevice(0, VExecTracer());
				tensors[it.first] = tensor;
			}
		}
	}

	return tensors;
}

VTensorList VUtils::toTensorList(VSession session, VList list) {
	VTensorList tensors;

	bool bCpuFound = false;
	bool bGpuFound = false;

	for (auto& it : list) {
		VTensor tensor(session, (VHTensor)(VHandle)it);
		tensors.push_back(tensor);
		if (tensor.device() < 0) bCpuFound = true;
		else bGpuFound = true;
	}

	if (bCpuFound && bGpuFound) {
		VTensorList mixedTensors = tensors;
		tensors.clear();
		for (auto& it : mixedTensors) {
			VTensor tensor = it;
			if (tensor.device() < 0) {
				tensor = tensor.toDevice(0, VExecTracer());
			}
			tensors.push_back(tensor);
		}
	}

	return tensors;
}

VTensorDict VUtils::toTensorDict(VSession session, VTensorMap map) {
	VTensorDict tensors;
	
	for (auto& it : map) {
		VTensor tensor = it.second;
		tensors[to_string(it.first)] = tensor;
	}

	return tensors;
}

VTensorDict VUtils::mergeTensorDict(VTensorDict dict1, VTensorDict dict2) {
	VTensorDict merged;

	for (auto& it : dict1) {
		VTensor tensor = it.second;
		merged[it.first] = tensor;
	}

	for (auto& it : dict2) {
		VTensor tensor = it.second;
		if (merged.find(it.first) == merged.end()) merged[it.first] = tensor;
		else VP_THROW(VERR_TENSOR_MERGE);
	}

	return merged;
}

VTensorDict VUtils::mergeTensorDict(VTensorDict dict1, VTensorDict dict2, string name1) {
	VTensorDict merged;

	for (auto& it : dict1) {
		VTensor tensor = it.second;
		if (it.first == "#") merged[name1] = tensor;
		else if (merged.find(it.first) == merged.end()) merged[it.first] = tensor;
		else VP_THROW(VERR_TENSOR_MERGE);
	}

	for (auto& it : dict2) {
		VTensor tensor = it.second;
		if (merged.find(it.first) == merged.end()) merged[it.first] = tensor;
		else VP_THROW(VERR_TENSOR_MERGE);
	}

	return merged;
}

VTensorDict VUtils::mergeTensorDict(VTensorDict dict1, VTensorDict dict2, string name1, string name2) {
	VTensorDict merged;

	if (name1 == name2) VP_THROW(VERR_TENSOR_MERGE);

	for (auto& it : dict1) {
		VTensor tensor = it.second;
		merged[name1 + ":" + it.first] = tensor;
	}

	for (auto& it : dict2) {
		VTensor tensor = it.second;
		merged[name2 + ":" + it.first] = tensor;
	}

	return merged;
}

VTensorDict VUtils::dictToDevice(VTensorDict dict, int nDevice, VExecTracer tracer) {
	VTensorDict result;
	for (auto& it : dict) {
		VTensor tensor = it.second;
		if (tensor.device() != nDevice) {
			tensor = tensor.toDevice(nDevice, tracer);
		}
		result[it.first] = tensor;
	}

	return result;
}

VTensorDict VUtils::toDevice(VTensorDict dict, int nDevice, VExecTracer tracer) {
	bool need_conv = false;

	for (auto& it : dict) {
		VTensor tensor = it.second;
		if (tensor.device() != nDevice) {
			need_conv = true;
			break;
		}
	}

	if (!need_conv) return dict;

	VTensorDict conv_dict = {};

	for (auto& it : dict) {
		VTensor tensor = it.second;
		if (tensor.device() != nDevice) {
			tensor = tensor.toDevice(nDevice, tracer);
		}
		conv_dict[it.first] = tensor;
	}

	return conv_dict;
}

VDict VUtils::toDictInternal(VTensorDict tensors) {
	VDict dict;

	for (auto& it : tensors) {
		dict[it.first] = it.second.cloneCore();
	}

	return dict;
}

VDict VUtils::toDictInternal(VTensorDict* pTensors, VList names) {
	VDict dict;

	for (int n = 0; n < names.size(); n++) {
		VTensorDict tensors = pTensors[n];
		VDict subdict;

		for (auto& it : tensors) {
			//subdict[it.first] = it.second.getCore();
			subdict[it.first] = it.second.cloneCore();
		}

		dict[names[n]] = subdict;
	}

	return dict;
}

void VUtils::freeDictInternal(VDict tensorDict) {
	for (auto& it1 : tensorDict) {
		VDict subdict = it1.second;
		;
		for (auto& it2 : subdict) {
			VObjCore* pCore = (VObjCore*)(it2.second);
			pCore->destroy();
		}
	}
}

VDict VUtils::toDictExternal(VTensorDict tensors) {
	VDict dict;

	for (auto& it : tensors) {
		dict[it.first] = it.second.cloneHandle();
	}

	return dict;
}

/*
VDict VUtils::toDictExternal(VTensorDict* pTensors, VList names) {
	VDict dict;

	for (int n = 0; n < names.size(); n++) {
		dict[names[n]] = toDictExternal(pTensors[n]);
	}

	return dict;
}
*/

VDict VUtils::toDictExternal(VModuleDict modules) {
	VDict dict;

	for (auto& it : modules) {
		dict[it.first] = it.second.cloneHandle();
	}

	return dict;
}

VList VUtils::toListExternal(VTensorList tensors) {
	VList list;

	for (auto& it : tensors) {
		list.push_back(it.cloneHandle());
	}

	return list;
}

VTensorMap VUtils::toTensorMap(VSession session, VMap map) {
	VTensorMap tensors;

	for (auto& it : map) {
		VTensor tensor(session, (VHTensor)(VHandle)it.second);
		tensors[it.first] = tensor;
	}

	return tensors;
}

VMap VUtils::toMapInternal(VTensorMap tensors) {
	VMap map;

	for (auto& it : tensors) {
		map[it.first] = it.second.cloneCore();
	}

	return map;
}

VMap VUtils::toMapExternal(VTensorMap tensors) {
	VMap map;

	for (auto& it : tensors) {
		map[it.first] = it.second.cloneHandle();
	}

	return map;
}

VList VUtils::toListExternal(VModuleList modules) {
	VList list;

	for (auto& it : modules) {
		list.push_back((int64)it.cloneHandle());
	}

	return list;
}

void VUtils::appendWithNamespace(VTensorDict& vars, VTensorDict preds, string nspace) {
	string prefix = nspace + "::";

	for (auto& it : preds) {
		vars[prefix + it.first] = it.second;
	}
}

VValue VUtils::copy(VValue value) {
	switch (value.type()) {
	case VValueType::list:
	{
		VList srcList = value;
		VList newList;

		for (auto& it : srcList) newList.push_back(copy(it));

		return newList;
	}
	case VValueType::dict:
	{
		VDict srcDict = value;
		VDict newDict;

		for (auto& it : srcDict) newDict[it.first] = copy(it.second);

		return newDict;
	}
	case VValueType::map:
	{
		VMap srcMap= value;
		VMap newMap;

		for (auto& it : srcMap) newMap[it.first] = copy(it.second);

		return newMap;
	}
	case VValueType::shape:
	{
		//hs.cho
		VShape temp = value;
		return temp.copy();
	}
	default:
		return value;
	}
}

VTensorDict VUtils::copy(VTensorDict xs) {	// capsule만 대체하는 shallow copy
	VTensorDict clone;
	for (auto& it : xs) {
		clone[it.first] = it.second;
	}
	return clone;
}

bool VUtils::isInt(string str) {

	const char* pstr = str.c_str();
	int len = (int)strlen(pstr);

	if (len == 0) return false;

	if (pstr[0] == '-') {
		pstr++;
		if (--len == 0) return false;
	}

	for (int n = 0; n < len; n++) {
		if (pstr[n] < '0' || pstr[n] > '9') return false;
	}

	return true;
}

bool VUtils::isFloat(string str) {
	const char* pstr = str.c_str();
	int len = (int)strlen(pstr);

	if (len == 0) return false;

	if (pstr[0] == '-') {
		pstr++;
		if (--len == 0) return false;
	}

	bool dot_found = false;

	for (int n = 0; n < len; n++) {
		if ((pstr[n] < '0' || pstr[n] > '9') && pstr[n] != '.') return false;
		if (pstr[n] == '.') {
			if (dot_found) return false;
			dot_found = true;
		}
	}

	return true;
}

VStrList VUtils::explode(string str, string delimeters) {
	VStrList result;

	char buffer[1024];
	char* context;

	if (strcpy_s(buffer, 1024, str.c_str())) TP_THROW(VERR_COPY_STRING);

	char* token = strtok_s(buffer, delimeters.c_str(), &context);

	while (token) {
		result.push_back(token);
		token = strtok_s(NULL, delimeters.c_str(), &context);
	}

	return result;
}

#ifdef KA_WINDOWS
FILE* VUtils::fopen(string filepath, string mode) {
	FILE* fid = NULL;

	std::replace(filepath.begin(), filepath.end(), '/', '\\');

	if (fopen_s(&fid, filepath.c_str(), mode.c_str()) != 0) {
		VP_THROW(VERR_FILE_OPEN);
	}
	return fid;
}
#else
FILE* VUtils::fopen(string filepath, string mode) {
	return ::fopen(filepath.c_str(), mode.c_str());
}
#endif

int VUtils::toPaddingMode(string padding_mode) {
	if (padding_mode == "zeros") return (int)PaddingMode::zeros;
	if (padding_mode == "reflect") return (int)PaddingMode::reflect;
	if (padding_mode == "replicate") return (int)PaddingMode::replicate;
	if (padding_mode == "circular") return (int)PaddingMode::circular;
	
	VP_THROW1(VERR_UNKNOWN_PADDING_MODE, padding_mode);
}

#ifdef FREE_ALL_NOMS
//hs.cho
// 추후 boost::filesystem  사용
#ifdef KA_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>

#define _mkdir(filepath)  mkdir(filepath, 0777)
//inline int localtime_s(struct tm *tmp, const time_t *timer){ struct tm* tmp2=localtime(timer); memcpy(tmp,tmp2,sizeof(*tmp2));return 0;}
inline struct tm* localtime_s(struct tm* tmp, const time_t* timer) { localtime_r(timer, tmp); return 0; }
#define strcpy_s(a,b,c) !strncpy(a,c,b)
#define strtok_s strtok_r
#endif

// Check the C++ compiler, STL version, and OS type.
#if (defined(_MSC_VER) && (_MSVC_LANG >= 201703L || _HAS_CXX17)) || (defined(__GNUC__) && (__cplusplus >= 201703L))
	// ISO C++17 Standard (/std:c++17 or -std=c++17)
#include <filesystem>
namespace fs = std::filesystem;
#else
//hs.cho
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem> // C++14
namespace fs = std::experimental::filesystem;
#endif



#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

VUtils kutil;

//hs.cho
//#ifdef KA_WINDOWS
void VUtils::mkdir(string path) {
#ifdef KA_WINDOWS
	std::replace(path.begin(), path.end(), '/', '\\');
#endif
	::_mkdir(path.c_str());
	//printf("\"%s\" has been created.\n", path.c_str());
}

KList VUtils::list_dir(string path) {
	KList list;

	//hs.cho
	//#ifdef KA_WINDOWS
	path = path + "/*";
#ifdef KA_WINDOWS
	std::replace(path.begin(), path.end(), '/', '\\');
#endif
	_finddata_t fd;
	intptr_t hObject;
	int64 result = 1;

	hObject = _findfirst(path.c_str(), &fd);

	if (hObject == -1) return list;

	while (result != -1) {
		if (fd.name[0] != '.') list.push_back(fd.name);
		result = _findnext(hObject, &fd);

	}
#ifdef NORANDOM
	std::sort(list.begin(), list.end(), [](const VValue& left, const VValue& right) {
		return (strcmp(left.desc().c_str(), right.desc().c_str()) < 0) ? true : false;
		});
#endif	
	_findclose(hObject);

	return list;
	//#else
	//	DIR* dir;
	//	struct dirent* ent;
	//	if ((dir = opendir(path.c_str())) != NULL) {
	//		while ((ent = readdir(dir)) != NULL) {
	//			if (ent->d_name[0] == '.') continue;
	//			list.push_back(ent->d_name);
	//			//clogger.Print("%s", ent->d_name);
	//		}
	//		closedir(dir);
	//	}
	//	else {
	//		throw KaiException(KERR_ASSERT);
	//	}
	//
	//	return list;
	//#endif
}

////hs.cho
////#ifdef KA_WINDOWS
//#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
//#include <experimental/filesystem> // C++14
//namespace fs = std::experimental::filesystem;

void VUtils::remove_all(string path) {
	fs::remove_all(path.c_str());
}
//#else
//bool VUtils::remove_all(string path) {
//	throw KaiException(KERR_UNIMPLEMENTED_YET);
//	return false;
//}
//#endif

string VUtils::get_timestamp(time_t tm) {
	struct tm timeinfo;
	char buffer[80];

	if (localtime_s(&timeinfo, &tm)) throw KaiException(KERR_FAILURE_ON_GET_LOCALTIME);

	strftime(buffer, 80, "%D %T", &timeinfo);
	return string(buffer);
}

string VUtils::get_date_8(time_t tm) {
	struct tm timeinfo;
	char buffer[80];

#ifdef KA_WINDOWS
	localtime_s(&timeinfo, &tm);
	struct tm* now = &timeinfo;
#else
	struct tm* now = localtime(&tm);
#endif

	snprintf(buffer, 80, "%04d%02d%02d", now->tm_year + 1900, now->tm_mon + 1, now->tm_mday);

	return string(buffer);
}

void VUtils::load_jpeg_image_pixels(float* pBuf, string filepath, KShape data_shape) {
	//hs.cho
#ifdef KA_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#endif
	cv::Mat img = cv::imread(filepath, 1);
	cv::resize(img, img, cv::Size((int)data_shape[0], (int)data_shape[1]), 0, 0, cv::INTER_AREA); //hs.cho cubic interpolation result can vary in some versions of OPENCV

	int chn = (int)img.channels();

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			cv::Vec3b intensity = img.at<cv::Vec3b>(j, i);
			for (int k = 0; k < chn; k++) {
				float dump = (float)intensity.val[k];;
				*pBuf++ = (float)intensity.val[k];
			}
		}
	}
}

vector<vector<string>> VUtils::load_csv(string filepath, KList* pHead) {
	//hs.cho
#ifdef KA_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#endif

	ifstream infile(filepath);

	if (infile.fail()) throw KaiException(KERR_FILE_OPEN_FAILURE, filepath);

	vector<vector<string>> rows;

	string line;
	char buffer[1024];
	char* context;

	getline(infile, line);
	if (pHead) {
		if (strcpy_s(buffer, 1024, line.c_str())) throw KaiException(KERR_ASSERT);
		char* token = strtok_s(buffer, ",", &context);
		while (token) {
			(*pHead).push_back(token);
			token = strtok_s(NULL, ",", &context);
		}
	}

	while (std::getline(infile, line)) {
		if (line[0] == ',') {
			line = "0" + line;
		}
		if (line[line.length() - 1] == ',') {
			line = line + "0";;
		}

		std::size_t pos = line.find(",,");
		while (pos != std::string::npos) {
			line = line.substr(0, pos + 1) + "0" + line.substr(pos + 1);
			pos = line.find(",,");
		}

		if (strcpy_s(buffer, 1024, line.c_str())) throw KaiException(KERR_ASSERT);
		char* token = strtok_s(buffer, ",", &context);
		vector<string> row;
		while (token) {
			row.push_back(token);
			token = strtok_s(NULL, ",", &context);
		}
		rows.push_back(row);
	}

	infile.close();

	return rows;
}

VValue VUtils::seek_dict_reset(VDict dict, string sKey, VValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) return kDefaultValue;
	else {
		VValue value = dict[sKey];
		dict.erase(sKey);
		return value;
	}
}

VValue VUtils::seek_set_dict(VDict& dict, string sKey, VValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) {
		if (kDefaultValue.is_none()) throw KaiException(KERR_NEED_PROPERTY_VALUE, sKey);
		dict[sKey] = kDefaultValue;
		return kDefaultValue;
	}
	else {
		return dict[sKey];
	}
}

void VUtils::save_value(FILE* fid, VValue value) {
	save_int64(fid, (int64)value.type());

	switch (value.type()) {
	case KeValueType::none:
		break;
	case KeValueType::int32:
		save_int32(fid, (int)value);
		break;
	case KeValueType::int64:
		save_int64(fid, (int64)value);
		break;
	case KeValueType::float32:
		save_float(fid, (float)value);
		break;
	case KeValueType::string:
		save_str(fid, (string)value);
		break;
	case KeValueType::list:
		save_list(fid, (KList)value);
		break;
	case KeValueType::dict:
		save_dict(fid, (VDict)value);
		break;
	case KeValueType::tensor:
		throw KaiException(KERR_UNIMPLEMENTED_YET);
		break;
	case KeValueType::object:
		throw KaiException(KERR_UNSUPPORTED_HANLE_TYPE_IN_SAVE);
		break;
#ifdef NOT_DEPRECIATED
		{
			throw KaiException(KERR_UNSUPPORTED_HANLE_TYPE_IN_SAVE);
			break;
			KHObject hObject = value;
			KeObjectType obj_type = hObject->get_type();
			save_int64(fid, (int64)obj_type);
			if (obj_type == KeObjectType::narray) {
				KNArr arr = NARRAY(value);
				save_array(fid, arr);
			}
			else if (obj_type == KeObjectType::farray) {
				KFArr arr = FARRAY(value);
				save_array(fid, arr);
			}
			else {
				throw KaiException(KERR_UNSUPPORTED_HANLE_TYPE_IN_SAVE);
			}
		}
#endif
	}
}

VValue VUtils::read_value(FILE* fid) {
	VValue value;
	enum class KeValueType type = (enum class KeValueType) read_int64(fid);

	switch (type) {
	case KeValueType::none:
		break;
	case KeValueType::int32:
		value = read_int32(fid);
		break;
	case KeValueType::int64:
		value = read_int64(fid);
		break;
	case KeValueType::float32:
		value = read_float(fid);
		break;
	case KeValueType::string:
		value = read_str(fid);
		break;
	case KeValueType::list:
		value = read_list(fid);
		break;
	case KeValueType::dict:
		value = read_dict(fid);
		break;
	case KeValueType::tensor:
		throw KaiException(KERR_UNIMPLEMENTED_YET);
		break;
	case KeValueType::object:
		throw KaiException(KERR_UNSUPPORTED_HANLE_TYPE_IN_SAVE);
		break;
#ifdef NOT_DEPRECIATED
	case KeValueType::object:
		obj_type = (enum class KeObjectType) read_int64(fid);
		switch (obj_type) {
		case KeObjectType::narray:
		{
			KNArr arr = read_narray(fid);
			value = arr.get_core();
			/*
			KNArr array = read_n64array(fid);
			KNArr* pArray = new KNArr();
			*pArray = array;
			value = KArray(array_type::at_int, (KHArray)pArray);
			*/
		}
		break;
		case KeObjectType::farray:
		{
			KFArr arr = read_farray(fid);
			value = arr.get_core();
		}
		break;
		}
#endif
	}

	return value;
}

void VUtils::save_int32(FILE* fid, int dat) {
	if (fwrite(&dat, sizeof(int), 1, fid) != 1) throw KaiException(KERR_FILE_SAVE_INT_FAILURE);
}

void VUtils::save_int64(FILE* fid, int64 dat) {
	if (fwrite(&dat, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_FILE_SAVE_INT_FAILURE);
}

void VUtils::save_float(FILE* fid, float dat) {
	if (fwrite(&dat, sizeof(float), 1, fid) != 1) throw KaiException(KERR_FILE_SAVE_FLOAT_FAILURE);
}

void VUtils::save_str(FILE* fid, string dat) {
	int64 length = (int64)dat.length();
	save_int64(fid, length);
	if (fwrite(dat.c_str(), sizeof(char), length, fid) != length) throw KaiException(KERR_FILE_SAVE_STRING_FAILURE);
}

void VUtils::save_list(FILE* fid, KList list) {
	save_int64(fid, list.size());
	for (auto it = list.begin(); it != list.end(); it++) {
		save_value(fid, *it);
	}
}

void VUtils::save_dict(FILE* fid, VDict dict) {
	save_int64(fid, dict.size());
	for (auto it = dict.begin(); it != dict.end(); it++) {
		save_str(fid, it->first);
		save_value(fid, it->second);
	}
}

void VUtils::save_shape(FILE* fid, KShape shape) {
	int64 dim = shape.size();
	save_int64(fid, dim);
	for (int64 n = 0; n < dim; n++) {
		save_int64(fid, shape[n]);
	}
}

int VUtils::read_int32(FILE* fid) {
	int dat;
	if (fread(&dat, sizeof(int), 1, fid) != 1) throw KaiException(KERR_FILE_READ_INT_FAILURE);
	return dat;
}

int64 VUtils::read_int64(FILE* fid) {
	int64 dat;
	if (fread(&dat, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_FILE_READ_INT_FAILURE);
	return dat;
}

float VUtils::read_float(FILE* fid) {
	float dat;
	if (fread(&dat, sizeof(float), 1, fid) != 1) throw KaiException(KERR_FILE_READ_FLOAT_FAILURE);
	return dat;
}

string VUtils::read_str(FILE* fid) {
	int64 length = read_int64(fid);
	char* piece = new char[length + 1];
	if (piece == NULL) VP_THROW(VERR_HOSTMEM_ALLOC_FAILURE);
	if (fread(piece, sizeof(char), length, fid) != length) throw KaiException(KERR_FILE_READ_STRING_FAILURE);
	piece[length] = 0;
	string str = piece;
	delete piece;
	return str;
}

KList VUtils::read_list(FILE* fid) {
	KList list;
	int64 size = read_int64(fid);
	for (int64 n = 0; n < size; n++) {
		VValue value = read_value(fid);
		list.push_back(value);
	}
	return list;
}

VDict VUtils::read_dict(FILE* fid) {
	VDict dict;
	int64 size = read_int64(fid);
	for (int64 n = 0; n < size; n++) {
		string key = read_str(fid);
		VValue value = read_value(fid);
		dict[key] = value;
	}
	return dict;
}

void VUtils::read_shape(FILE* fid, KShape& shape) {
	int64 dim = read_int64(fid);
	int64 ax_size[KA_MAX_DIM];

	if (fread(ax_size, sizeof(int64), dim, fid) != dim) throw KaiException(KERR_ASSERT);

	shape = KShape(dim, ax_size);
}

#ifdef NOT_DEPRECIATED
template <class T> void VUtils::save_array(FILE* fid, KaiArray<T> arr) {
	save_shape(fid, arr.shape());
	arr = arr.to_host();
	int64 size = arr.total_size();
	T* p = arr.data_ptr();
	if (fwrite(p, sizeof(T), size, fid) != size) throw KaiException(KERR_SAVE_ARRAY_FAILURE);
}

template void VUtils::save_array(FILE* fid, KNArr arr);
template void VUtils::save_array(FILE* fid, KFArr arr);

KNArr VUtils::read_narray(FILE* fid) {
	KShape shape;
	read_shape(fid, shape);
	KNArr arr(shape);
	int64 size = arr.total_size();
	int64* p = arr.data_ptr();
	if (fread(p, sizeof(int64), size, fid) != size) throw KaiException(KERR_READ_ARRAY_FAILURE);
	return arr;
}

KFArr VUtils::read_farray(FILE* fid) {
	KShape shape;
	read_shape(fid, shape);
	KFArr arr(shape);
	int64 size = arr.total_size();
	float* p = arr.data_ptr();
	if (fread(p, sizeof(float), size, fid) != size) throw KaiException(KERR_READ_ARRAY_FAILURE);
	return arr;
}
#endif

#ifdef TO_SUPPRT_VERSION_2021
KFArr VUtils::get_host_param(KaiMath* pMath, VDict pm_dict, string sKey, string sSubkey) {
	VDict param = pm_dict[sKey];
	VDict pm = param[sSubkey];
	KFArr arr = FARRAY(pm["_pm_"]);
	return pMath->to_host(arr);
}
#endif

VValue VUtils::deep_copy(VValue value) {
	switch (value.type()) {
	case KeValueType::list:
	{
		KList srcList = value;
		KList newList;

		for (auto& it : srcList) newList.push_back(deep_copy(it));

		return newList;
	}
	case KeValueType::dict:
	{
		VDict srcDict = value;
		VDict newDict;

		for (auto& it : srcDict) newDict[it.first] = deep_copy(it.second);

		return newDict;
	}
	case KeValueType::tensor:
	{
		VValue clone;
		KCRetCode retCode = KaiCoreCopyTensor(&clone, value);
		if (retCode != 0) throw KaiException(retCode);

		return clone;
	}
	case KeValueType::shape:
		return ((KShape)value).copy();
	default:
		return value;
	}
}

VValue VUtils::upload_array(VDict info) {
	if ((string)info["type"] == "float") {
		throw KaiException(KERR_UNIMPLEMENTED_YET);
		// core 측 텐서 연산 기능 이용하도록 수정
		/*
		KTensorMath hmath;

		KShape shape = info["shape"];
		KTensor tensor = hmath.allocHostFloat(shape);

		float* pDest = tensor.fdata_ptr();
		float* pSrc = (float*)info["data"];

		memcpy(pDest, pSrc, sizeof(float) * shape.total_size());

		return tensor;
		*/
	}
	else {
		throw KaiException(KERR_UNIMPLEMENTED_YET);
	}
}

void VUtils::pretty_dump(VDict dict, string sTitle, int depth) {
	printf("%*s%s[dict]:\n", depth++ * 2, "", sTitle.c_str());

	for (auto& it : dict) {
		switch (it.second.type()) {
		case KeValueType::dict:
			pretty_dump((VDict)it.second, it.first, depth);
			break;
		case KeValueType::list:
			pretty_dump((KList)it.second, it.first, depth);
			break;
		case KeValueType::object:
			printf("%*s%s[object]\n", depth * 2, "", it.first.c_str());
			break;
		default:
			printf("%*s%s: %s\n", depth * 2, "", it.first.c_str(), it.second.desc().c_str());
			break;
		}
	}
}

void VUtils::pretty_dump(KList list, string sTitle, int depth) {
	printf("%*s%s[list]:\n", depth++ * 2, "", sTitle.c_str());

	int nth = 0;

	for (auto& it : list) {
		string sSubTitle = "[" + std::to_string(nth) + "]";
		switch (it.type()) {
		case KeValueType::dict:
			pretty_dump((VDict)it, sSubTitle, depth);
			break;
		case KeValueType::list:
			pretty_dump((KList)it, sSubTitle, depth);
			break;
		case KeValueType::object:
			printf("%*s%s[object]\n", depth * 2, "", sSubTitle.c_str());
			break;
		default:
			printf("%*s%s: %s\n", depth * 2, "", sSubTitle.c_str(), it.desc().c_str());
			break;
		}
		nth++;
	}
}

int64 VUtils::conv_to_actfunc_id(string funcname) {
	if (funcname == "none" || funcname == "linear" || funcname == "") return (int64)KeActFunc::none;
	else if (funcname == "relu") return (int64)KeActFunc::relu;
	else if (funcname == "sigmoid") return (int64)KeActFunc::sigmoid;
	else if (funcname == "tanh") return (int64)KeActFunc::tanh;
	else if (funcname == "leaky" || funcname == "leaky_relu") return (int64)KeActFunc::leaky_relu;
	else if (funcname == "gelu") return (int64)KeActFunc::gelu;
	else if (funcname == "selu") return (int64)KeActFunc::selu;
	else if (funcname == "custom") return (int64)KeActFunc::custom;
	else if (funcname == "mish") return (int64)KeActFunc::mish;
	else if (funcname == "swish") return (int64)KeActFunc::swish;

	throw KaiException(KERR_UNKNOWN_ACTFUNCNAME, funcname);
}

bool VUtils::isMember(KStrList slList, string sData) {
	return (std::find(slList.begin(), slList.end(), sData) != slList.end());
}

// core 측 텐서 연산 기능 이용하도록 수정
/*
int64 VUtils::nan_check(KTensor tensor, bool bThrow) {
	KTensorMath hostmath;
	KTensor host = tensor.to_host();
	int64 size = host.total_size();
	float* fp = host.fdata_ptr();

	for (int64 n = 0; n < size; n++) {
		if (isnan(fp[n]) || isinf(fp[n]) || isinf(-fp[n])) {
			if (bThrow) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);
			return n;
		}
	}

	return -1;
}
*/

int VUtils::getListIndex(string str, KStrList noms, int errCode) {
	for (int n = 0; n < (int)noms.size(); n++) {
		if (str == noms[n]) return n;
	}
	if (errCode != 0) throw KaiException(errCode);
	return -1;
}
#endif
