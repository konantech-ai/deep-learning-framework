#pragma once

#include "../include/vapi.h"

typedef map<string, VTensor> VTensorDict;
typedef map<string, VModule> VModuleDict;

typedef map<int, VTensor> VTensorMap;

typedef vector<VTensor> VTensorList;
typedef vector<VModule> VModuleList;

class VExecTracer;

class VUtils {
public:
	VValue seek_dict(VDict dict, string sKey, VValue kDefaultValue);
	VValue seek_dict_set(VDict dict, string sKey, VValue kDefaultValue);
	VValue seek_dict_reset(VDict dict, string sKey, VValue kDefaultValue);

	string tolower(string str);

	VTensorDict toTensorDict(VSession session, VDict dict);
	VTensorDict toTensorDict(VSession session, VTensorMap tensors);
	VTensorDict mergeTensorDict(VTensorDict dict1, VTensorDict dict2);
	VTensorDict mergeTensorDict(VTensorDict dict1, VTensorDict dict2, string name1);
	VTensorDict mergeTensorDict(VTensorDict dict1, VTensorDict dict2, string name1, string name2);
	VTensorDict dictToDevice(VTensorDict dict, int nDevice, VExecTracer tracer);

	VTensorDict toDevice(VTensorDict dict, int nDevice, VExecTracer tracer);

	VDict toDictInternal(VTensorDict tensors);
	VDict toDictExternal(VTensorDict tensors);
	VDict toDictExternal(VModuleDict modules);

	VList toListExternal(VTensorList modules);
	VList toListExternal(VModuleList modules);

	VDict toDictInternal(VTensorDict* pTensors, VList names);
	void freeDictInternal(VDict tensorDict);
	//VDict toDictExternal(VTensorDict* pTensors, VList names);

	VTensorList toTensorList(VSession session, VList list);

	VTensorMap toTensorMap(VSession session, VMap dict);
	VMap toMapInternal(VTensorMap tensors);
	VMap toMapExternal(VTensorMap tensors);

	void appendWithNamespace(VTensorDict& vars, VTensorDict preds, string nspace);

	VValue copy(VValue value);	// list, dict 구조만 deep-copy, array는 공유 처리
	VTensorDict copy(VTensorDict xs);	// capsule만 대체하는 shallow copy

	bool isInt(string str);
	bool isFloat(string str);

	FILE* fopen(string filepath, string mode);

	VStrList explode(string str, string delimeters =" \t\r\n,");

	int toPaddingMode(string padding_mode);

#ifdef FREE_ALL_NOMS
	void mkdir(string dir);
	void remove_all(string sFilePath);

	KList list_dir(string path);

	string get_timestamp(time_t tm);
	string get_date_8(time_t tm);

	void load_jpeg_image_pixels(float* pBuf, string filepath, KShape data_shape);
	vector<vector<string>> load_csv(string filepath, KList* pHead = NULL);

	VValue seek_dict_reset(VDict dict, string sKey, VValue kDefaultValue);
	VValue seek_set_dict(VDict& dict, string sKey, VValue kDefaultValue = VValue());

	void save_value(FILE* fid, VValue value);
	VValue read_value(FILE* fid);

	void save_int32(FILE* fid, int dat);
	void save_int64(FILE* fid, int64 dat);
	void save_str(FILE* fid, string dat);
	void save_float(FILE* fid, float dat);

	void save_list(FILE* fid, KList value);
	void save_dict(FILE* fid, VDict value);

	void save_shape(FILE* fid, KShape shape);

	int read_int32(FILE* fid);
	int64 read_int64(FILE* fid);
	string read_str(FILE* fid);
	float read_float(FILE* fid);

	KList read_list(FILE* fid);
	VDict read_dict(FILE* fid);

	void read_shape(FILE* fid, KShape& shape);

#ifdef NOT_DEPRECIATED
	KNArr read_narray(FILE* fid);
	KFArr read_farray(FILE* fid);

	template <class T> void save_array(FILE* fid, KaiArray<T> arr);
#endif

#ifdef TO_SUPPRT_VERSION_2021
	KFArr get_host_param(KaiMath* pMath, VDict pm_dict, string sKey, string sSubkey);
#endif

	VValue copy(VValue value);	// list, dict 구조만 deep-copy, array는 공유 처리
	VValue deep_copy(VValue value);	// list, dict 구조 및 array 모두 deep-copy

	VValue upload_array(VDict info);

	void pretty_dump(VDict dict, string sTitle, int depth = 0);
	void pretty_dump(KList list, string sTitle, int depth = 0);

	int64 nan_check(KTensor tensor, bool bThrow = true);

	int64 conv_to_actfunc_id(string funcname);

	bool isMember(KStrList slList, string sData);

	int getListIndex(string str, KStrList noms, int errCode);
#endif
};

extern VUtils vutils;