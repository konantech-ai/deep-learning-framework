#pragma once

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "../utils/tp_common.h"
#include "../utils/tp_stream.h"

struct WaveInfoHeader {
public:
	char ChunkID[4], Format[4], Subchunk1ID[4], Subchunk2ID[4];
	int ChunkSize, Subchunk1Size, SampleRate, ByteRate, Subchunk2Size, Subchunk2Offset;
	short AudioFormat, NumChannels, BlockAlign, BitsPerSample;
};

class TpUtils {
public:
	static string join(VStrList list, string sDelimeter);
	static string join_dict_names(VDict dict, string sDelimeter);
	static string join_dict_values(VDict dict, string sDelimeter);
	static string join(VList list, string sDelimeter);
	static string getcwd();
	static void mkdir(string path);
	static void mkdir(string basepath, string subpath);
	static VStrList list_dir(string path, string filter = "*");
	static VStrList strtok(string str, string delimeters=" \t\r\n,");
	static VStrList read_file_lines(string filePath);

	static VList strtokToList(string str, string delimeters = " \t\r\n,");
	static bool file_exist(string path);
	static string getFileExt(string filename);
	static FILE* fopen(string filepath, string mode, bool bthrow = true);
	static void read_wav_file_header(string filepath, WaveInfoHeader* pInfo);
	static cv::Mat load_image(string filepath);
	static cv::Mat resize_image(cv::Mat mat, int nMaxSize);
	static void load_jpeg_image_pixels(float* pBuf, string filepath, VShape data_shape, bool crop);
	static string get_timestamp(time_t time);
	static string get_date_8(time_t time);
	static string get_time_6(time_t time);
	static void dumpStrList(string sTitle, VStrList slList);
	static 	string tolower(string str);
	static VValue seekDict(VDict dict, string sKey, VValue kDefaultValue);
	static VDict mergeDict(VDict dict1, VDict dict2);	// 중복된 키에 대해서는 dict1 우선임

	static inline void ltrim(string& s) {
		s.erase(s.begin(), find_if(s.begin(), s.end(), not1(ptr_fun(isspace))));
	}

	static inline void rtrim(string& s) {
		s.erase(find_if(s.rbegin(), s.rend(), not1(ptr_fun(isspace))).base(), s.end());
	}

	static inline void trim(std::string& s) {
		ltrim(s);
		rtrim(s);
	}

	static int find(VList list, VValue element);

	static cv::Mat drawImageGrid(ETensorList tensors, string image_shape);
	static cv::Mat createCanvas(int height, int width, int chns);

	static void displayImage(string name, cv::Mat mat, int milliSec);
	static void drawTensorImage(cv::Mat mat, ETensor tensor, cv::Rect rect, int dataIdx);
	static void addMapImage(cv::Mat mat, ETensor tensor, cv::Rect rect, int ndat, int nchn, int color_flags);
	static void addFilterImage(cv::Mat mat, ETensor tensor, cv::Rect rect, int xchn, int ychn, int color_flags, float coef=1.0f);
	static void copyCanvas(cv::Mat dstMat, cv::Mat srcMat);
	static void saveImage(string path, string name, cv::Mat mat);

	static string toJsonString(VDict dict);
	static string toJsonString(VList list);

	static string to_string(VDataType type);
	static VDataType to_data_type(string type);
	static int byte_size(VDataType type);

	static VDict TensorDictToDict(ETensorDict dict, bool upload);
	static VList TensorListToList(ETensorList dict, bool upload);
	static VDict LossDictToDict(ELossDict dict);
	static ETensorDict DictToTensorDict(ENN nn, VDict handles);
	static ETensorDicts DictToTensorDicts(ENN nn, VDict handles);
	static EModuleDict DictToModuleDict(ENN nn, VDict handles);
	static EModuleList ListToModuleList(ENN nn, VList handles);
	static ETensorList ListToTensorList(ENN nn, VList handles, bool download);
	static VStrList ListToStrList(VList list);

	static string id_to_token(int64 id);
	static int64 token_to_id(string token);

	static VValue deep_copy(VValue value);	// list, dict 구조 및 array 모두 deep-copy

	static void FileDump(string filepath);

	static ETensorDict LoadNPZ(ENN nn, string filepath);

	static void ms_dump_arr_feat2(int nth, const char* sTitle, int size, int h, int w, int c, float* parr);
	static void ms_dump_arr_feat3(int nth, const char* sTitle, int size, int h, int w, int c, unsigned char* parr);
	static void ms_dump_arr_feat4(int nth, const char* sTitle, int size, int h, int w, int c, unsigned char* parr);

	static float ms_get_pixel(cv::Mat img, int r, int c, int k);
	static float ms_get_pixel(float* pBuf, int cols, int chns, int r, int c, int k);
	static void ms_set_pixel(float* pBuf, int cols, int chns, int r, int c, int k, float val);
	static void ms_add_pixel(float* pBuf, int cols, int chns, int r, int c, int k, float val);

	static ETensor ms_loadNumpyArray(ENN nn, FILE* fin);
};
