#include "../utils/tp_common.h"
#include "../utils/tp_utils.h"
#include "../connect/tp_api_conn.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_module.h"
#include "../objects/tp_nn.h"
#include "../utils/tp_exception.h"
#include "../utils/tp_json_parser.h"
#include <algorithm>
#include <memory>
#include <string>
#include <stdexcept>

#ifdef KA_WINDOWS
#include <direct.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#define _mkdir(filepath)  mkdir(filepath, 0777)
#endif


#ifdef FOR_LINUX
#define strcpy_s(a,b,c) !strncpy(a,c,b)
#define strtok_s strtok_r
#define _atoi64(a)  strtoll(a,NULL, 10)
inline struct tm* localtime_s(struct tm* tmp, const time_t* timer) { localtime_r(timer, tmp); return 0; }
#endif


string TpUtils::join(VStrList list, string sDelimeter) {
	if (list.size() == 0) return "";

	string sJoined = list[0];

	for (size_t n = 1; n < list.size(); n++) {
		sJoined += sDelimeter;
		//hs.cho
		//sJoined += list[n];
		std::string temp = list[n];
		sJoined += temp;
	}

	return sJoined;
}

string TpUtils::join_dict_names(VDict dict, string sDelimeter) {
	if (dict.size() == 0) return "";

	string sJoined = "";
	string sPrefix = "";

	for (auto& it : dict) {
		sJoined += sPrefix + it.first;
		sPrefix = sDelimeter;
	}

	return sJoined;
}

string TpUtils::join_dict_values(VDict dict, string sDelimeter) {
	if (dict.size() == 0) return "";

	string sJoined = "";
	string sPrefix = "";

	for (auto& it : dict) {
		string value = it.second.desc();
		sJoined += sPrefix + value;
		sPrefix = sDelimeter;
	}

	return sJoined;
}

string TpUtils::join(VList list, string sDelimeter) {
	if (list.size() == 0) return "";

	string sJoined = "";
	string sPrefix = "";

	for (auto& it : list) {
		string value = it.desc();
		sJoined += sPrefix + value;
		sPrefix = sDelimeter;
	}

	return sJoined;
}

string TpUtils::getcwd() {
	char buff[1024];
#ifdef FOR_LINUX
	::getcwd(buff, 1024);
#else
	::_getcwd(buff, 1024);
#endif
	string cwd(buff);
	return cwd;
}

void TpUtils::mkdir(string path) {
#ifdef KA_WINDOWS
	std::replace(path.begin(), path.end(), '/', '\\');
#endif
	int rs = ::_mkdir(path.c_str());
}

void TpUtils::mkdir(string basepath, string subpath) {
#ifdef KA_WINDOWS
	std::replace(basepath.begin(), basepath.end(), '/', '\\');
	std::replace(subpath.begin(), subpath.end(), '/', '\\');
#endif
	VStrList folders = strtok(subpath, "\\");

	for (auto& it : folders) {
#ifdef KA_WINDOWS
		basepath = basepath + it + "\\";
#else
		basepath = basepath + it + "/";
#endif
		::_mkdir(basepath.c_str());
	}
}

VStrList TpUtils::list_dir(string path, string filter) {
	VStrList list;

	//hs.cho
	//#ifdef KA_WINDOWS
	path = path + "/" + filter;
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
}

VStrList TpUtils::strtok(string str, string delimeters) {
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

VList TpUtils::strtokToList(string str, string delimeters) {
	VList result;

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

bool TpUtils::file_exist(string filepath) {
	FILE* fid = fopen(filepath, "r", false);
	if (fid == NULL) return false;
	fclose(fid);
	return true;
}

string TpUtils::getFileExt(string filename) {
	size_t pos = filename.find_last_of(".");
	if (pos == string::npos) return "";
	return filename.substr(pos + 1);
}

#ifdef KA_WINDOWS
FILE* TpUtils::fopen(string filepath, string mode, bool bThrow) {
	FILE* fid = NULL;

	std::replace(filepath.begin(), filepath.end(), '/', '\\');

	// Delete : -->
	// [Opinion]
	// Basically wrong 'Path or FileName' is user-error 
	// Below 'ErrorCode' will address user-error
	/*
	size_t len = filepath.length();
	if (filepath[len - 1] == '\\') filepath = filepath.substr(0, len - 1);

	size_t pos;
	while ((pos = filepath.find("\\\\")) != std::string::npos) {
		filepath = filepath.substr(0, pos) + filepath.substr(pos + 2);
	}
	*/
	// Delete : <--

	int64 ErrorCode = fopen_s(&fid, filepath.c_str(), mode.c_str());
	if (ErrorCode != 0) {
		if (bThrow) {
			print("filepath: %s, ErrorCode %lld", filepath.c_str(), ErrorCode);
			TP_THROW(VERR_FILE_OPEN);
		}
		else return NULL;
	}
	return fid;
}
#else
FILE* TpUtils::fopen(string filepath, string mode, bool bThrow) {
	size_t len = filepath.length();
	if (filepath[len - 1] == '\\') filepath = filepath.substr(0, len - 1);
	FILE* fid = ::fopen(filepath.c_str(), mode.c_str());
	if (fid == NULL && bThrow) TP_THROW(VERR_FILE_OPEN);
	return fid;
}
#endif

VStrList TpUtils::read_file_lines(string filePath) {
	VStrList lines;

	FILE* fid = fopen(filePath.c_str(), "rt");

	char buffer[10240];

	while (true) {
		if (fgets(buffer, 10240, fid) == NULL) break;
		buffer[strlen(buffer) - 1] = 0;
		lines.push_back(string(buffer));
	}

	fclose(fid);

	return lines;
}

void TpUtils::read_wav_file_header(string filepath, WaveInfoHeader* pInfo)
{
	string ext = filepath.substr(filepath.find_last_of('.') + 1);

	if (ext == "wav") {
		// Read the wave file
		FILE* fhandle = NULL;
		
		try {
			fhandle = TpUtils::fopen(filepath.c_str(), "rb");

			fread(pInfo->ChunkID, 1, 4, fhandle);
			fread(&pInfo->ChunkSize, 4, 1, fhandle);
			fread(pInfo->Format, 1, 4, fhandle);
			fread(pInfo->Subchunk1ID, 1, 4, fhandle);
			fread(&pInfo->Subchunk1Size, 4, 1, fhandle);
			fread(&pInfo->AudioFormat, 2, 1, fhandle);
			fread(&pInfo->NumChannels, 2, 1, fhandle);
			fread(&pInfo->SampleRate, 4, 1, fhandle);
			fread(&pInfo->ByteRate, 4, 1, fhandle);
			fread(&pInfo->BlockAlign, 2, 1, fhandle);
			fread(&pInfo->BitsPerSample, 2, 1, fhandle);
			fread(&pInfo->Subchunk2ID, 1, 4, fhandle);

			if (strncmp(pInfo->ChunkID, "RIFF", 4) != 0) TP_THROW(VERR_CONTENT_HEADER);
			if (strncmp(pInfo->Format, "WAVE", 4) != 0) TP_THROW(VERR_CONTENT_HEADER);
			if (strncmp(pInfo->Subchunk1ID, "fmt", 3) != 0) TP_THROW(VERR_CONTENT_HEADER);
			if (pInfo->AudioFormat != 1) TP_THROW(VERR_CONTENT_HEADER);
			if (pInfo->NumChannels != 2) TP_THROW(VERR_CONTENT_HEADER);
			if (pInfo->SampleRate != 44100) TP_THROW(VERR_CONTENT_HEADER);
			if (pInfo->ByteRate != 176400) TP_THROW(VERR_CONTENT_HEADER);
			if (pInfo->BlockAlign != 4) TP_THROW(VERR_CONTENT_HEADER);
			if (pInfo->BitsPerSample != 16) TP_THROW(VERR_CONTENT_HEADER);

			while (strncmp(pInfo->Subchunk2ID, "data", 4) != 0) {
				fread(&pInfo->Subchunk2Size, 4, 1, fhandle);
				fseek(fhandle, pInfo->Subchunk2Size, SEEK_CUR);
				fread(&pInfo->Subchunk2ID, 1, 4, fhandle);
			}

			fread(&pInfo->Subchunk2Size, 4, 1, fhandle);

			int data_from = ftell(fhandle);
			fseek(fhandle, 0, SEEK_END);
			int data_end = ftell(fhandle);

			if (pInfo->Subchunk2Size != data_end - data_from) TP_THROW(VERR_CONTENT_HEADER);

			pInfo->Subchunk2Offset = data_from;

			/*
			if (bReadData) {
				delete[] pInfo->pData;
				pInfo->pData = new unsigned char[pInfo->Subchunk2Size]; // Create an element for every sample
				int64 nRead = fread(pInfo->pData, 1, pInfo->Subchunk2Size, fhandle); // Reading raw audio data
				if (nRead != pInfo->Subchunk2Size) throw VERR_ASSERT;
				//if (!feof(fhandle)) throw KaiException(VERR_ASSERT);
			}
			*/

			fclose(fhandle);

			fhandle = NULL;
		}
		catch (...) {
			if (fhandle) fclose(fhandle);
			TP_THROW(VERR_CONTENT_HEADER);
		}
	}
	else {
		TP_THROW(VERR_FILE_NAME);
	}
}

void TpUtils::ms_dump_arr_feat2(int nth, const char* sTitle, int size, int h, int w, int c, float* parr) {
	float first, last, min, max, avg, std;
	float sum = 0, sqsum = 0;

	int minpos = 0, maxpos = 0;

	float* pBuffer = parr; // (float*)malloc(sizeof(float) * size);

	first = min = max = pBuffer[0];
	last = pBuffer[size - 1];

	for (int k = 0; k < c; k++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int my_pos = y * w * c + x * c + k;
				int darknet_pos = k * h * w + y * w + x;
				float elem = pBuffer[my_pos];
				if (elem > max) {
					max = elem;
					maxpos = darknet_pos;
				}
				if (elem < min) {
					min = elem;
					minpos = darknet_pos;
				}
				sum += elem;
				sqsum += elem * elem;
			}
		}
	}

	avg = sum / (float)size;
	std = sqrtf(sqsum / size - avg * avg);

	print("%d\t%s\t[%d,%d,%d]\tfirst:%g\tlast:%g\tmin:%g(%d)\tmax:%g(%d)\tavg:%g\tstd:%g", nth, sTitle, h, w, c, first, last, min, minpos, max, maxpos, avg, std);

	int histogram[11];
	int nOut = 0;
	memset(histogram, 0, 11 * sizeof(int));

	if (min < max) {
		for (int64 n = 0; n < size; n++) {
			float elem = pBuffer[n];
			int nth = (int)((elem - min) * 10 / (max - min));
			if (nth >= 0 && nth <= 10) histogram[nth]++;
			else nOut++;
		}
	}
	print("\t\t\tHistogram:");
	for (int64 n = 0; n < 10; n++) {
		print(" %d", histogram[n]);
	}
	print(" (out:%d)", nOut);
	//delete[] pBuffer;
}

void TpUtils::ms_dump_arr_feat3(int nth, const char* sTitle, int size, int h, int w, int c, unsigned char* parr) {
	float first, last, min, max, avg, std;
	float sum = 0, sqsum = 0;

	int minpos = 0, maxpos = 0;

	first = min = max = (float)parr[0] / 255.0f;
	last = (float)parr[size - 1] / 255.0f;

	for (int n = 0; n < size; n++) {
		float elem = (float)parr[n] / 255.0f;

		if (elem > max) {
			max = elem;
			maxpos = n;
		}
		if (elem < min) {
			min = elem;
			minpos = n;
		}
		sum += elem;
		sqsum += elem * elem;
	}
	avg = sum / (float)size;
	std = sqrt(sqsum / size - avg * avg);

	//free(pBuffer);

	print("%d\t%s\t[%d,%d,%d]\tfirst:%g\tlast:%g\tmin:%g(%d)\tmax:%g(%d)\tavg:%g\tstd:%g", nth, sTitle, h, w, c, first, last, min, minpos, max, maxpos, avg, std);

	int histogram[11];
	memset(histogram, 0, 11 * sizeof(int));

	if (min < max) {
		for (int64 n = 0; n < size; n++) {
			float elem = (float)parr[n] / 255;
			int nth = (int)((elem - min) * 10 / (max - min));
			histogram[nth]++;
		}
	}
	print("\t\t\tHistogram:");
	for (int64 n = 0; n < 10; n++) {
		print(" %d", histogram[n]);
	}
	print("");
	//delete[] pBuffer;
}

void TpUtils::ms_dump_arr_feat4(int nth, const char* sTitle, int size, int h, int w, int c, unsigned char* parr) {
	int64 first, last, min, max;
	int64 sum = 0, sqsum = 0;
	double avg, std;

	int minpos = 0, maxpos = 0;

	first = min = max = parr[0];
	last = parr[size - 1];

	for (int n = 0; n < size; n++) {
		int elem = parr[n];

		if (elem > max) {
			max = elem;
			maxpos = n;
		}
		if (elem < min) {
			min = elem;
			minpos = n;
		}
		sum += elem;
		sqsum += elem * elem;
	}
	avg = (double)sum / (float)size;
	std = sqrt((double)sqsum / size - avg * avg);

	//free(pBuffer);

	print("%d\t%s\t[%d,%d,%d]\tfirst:%lld\tlast:%lld\tsum:%lld\tsqsum:%lld\tmin:%lld(%d)\tmax:%lld(%d)\tavg:%8.6f\tstd:%8.6f",
		nth, sTitle, h, w, c, first, last, sum, sqsum, min, minpos, max, maxpos, avg, std);

	int histogram[256];
	memset(histogram, 0, 256 * sizeof(int));

	for (int64 n = 0; n < size; n++) {
		int elem = parr[n];
		histogram[elem]++;
	}

	print("\t\t\tHistogram:");
	for (int64 n = 0; n < 256; n++) {
		print(" %d", histogram[n]);
		if (n % 16 == 15) print("");
	}
	print("");
	//delete[] pBuffer;
}

float TpUtils::ms_get_pixel(cv::Mat img, int r, int c, int k) {
	cv::Vec3b intensity = img.at<cv::Vec3b>(r, c);
	return intensity.val[k] / (float)255.;
}

float TpUtils::ms_get_pixel(float* pBuf, int cols, int chns, int r, int c, int k) {
	int idx = (r * cols + c) * chns + k;
	return pBuf[idx];
}

void TpUtils::ms_set_pixel(float* pBuf, int cols, int chns, int r, int c, int k, float val) {
	int idx = (r * cols + c) * chns + k;
	pBuf[idx] = val;
}

void TpUtils::ms_add_pixel(float* pBuf, int cols, int chns, int r, int c, int k, float val) {
	int idx = (r * cols + c) * chns + k;
	pBuf[idx] += val;
}

cv::Mat TpUtils::load_image(string filepath) {
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
	cv::Mat mat = cv::imread(filepath, cv::IMREAD_COLOR);
	return mat;
}

cv::Mat TpUtils::resize_image(cv::Mat mat, int nMaxSize) {
	cv::Mat dst;

	float wratio = (float)mat.cols / nMaxSize;
	float hratio = (float)mat.rows / nMaxSize;

	if (wratio > hratio) {
		int rows = (int)(nMaxSize * hratio / wratio);
		cv::resize(mat, dst, cv::Size(nMaxSize, rows));
	}
	else {
		int cols = (int)(nMaxSize * wratio / hratio);
		cv::resize(mat, dst, cv::Size(cols, nMaxSize));
	}

	return dst;
}

void TpUtils::load_jpeg_image_pixels(float* pImageBuf, string filepath, VShape data_shape, bool crop) {
	std::replace(filepath.begin(), filepath.end(), '/', '\\');

	cv::Mat mat = cv::imread(filepath, cv::IMREAD_COLOR);
	cv::Mat img;

	if (0) {
		ms_dump_arr_feat4(0, "direct read", mat.rows * mat.cols * mat.channels(), mat.rows, mat.cols, mat.channels(), mat.data);
		ms_dump_arr_feat3(0, "direct read", mat.rows * mat.cols * mat.channels(), mat.rows, mat.cols, mat.channels(), mat.data);
	}

	// 잘못된 패스나 파일 형식으로 인해 mat이 제대로 안 읽힌 경우의 예외 처리 방법을 조사해 반영할 것
	int channels = mat.channels();

	if (channels == 3) cv::cvtColor(mat, img, cv::COLOR_RGB2BGR);
	else if (channels == 4) cv::cvtColor(mat, img, cv::COLOR_RGBA2BGRA);

	if (0) {
		ms_dump_arr_feat3(0, "after convert", img.rows * img.cols * img.channels(), img.rows, img.cols, img.channels(), img.data);
	}

	int height = (int)data_shape[0];
	int width = (int)data_shape[1];

	int h_base = 0, w_base = 0;

	int r, c, k;
	int chns = img.channels();

	float w_scale = (float)(img.cols - 1) / (width - 1);
	float h_scale = (float)(img.rows - 1) / (height - 1);

	float* pTempBuf = new float[img.rows * width * chns];
	if (pTempBuf == NULL) TP_THROW2(VERR_HOSTMEM_ALLOC_FAILURE, std::to_string(img.rows * width * chns));

	for (k = 0; k < chns; ++k) {
		for (r = 0; r < img.rows; ++r) {
			for (c = 0; c < width; ++c) {
				float val = 0;
				if (c == width - 1 || img.cols == 1) {
					val = ms_get_pixel(img, r, img.cols - 1, k);
				}
				else {
					float sx = c * w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * ms_get_pixel(img, r, ix, k) + dx * ms_get_pixel(img, r, ix + 1, k);
				}
				ms_set_pixel(pTempBuf, width, chns, r, c, k, val);
			}
		}
	}

	for (k = 0; k < chns; ++k) {
		for (r = 0; r < height; ++r) {
			float sr = r * h_scale;
			int ir = (int)sr;
			float dr = sr - ir;
			for (c = 0; c < width; ++c) {
				float val = (1 - dr) * ms_get_pixel(pTempBuf, width, chns, ir, c, k);
				ms_set_pixel(pImageBuf, width, chns, r, c, k, val);
			}
			if (r == height - 1 || img.rows == 1) continue;
			for (c = 0; c < width; ++c) {
				float val = dr * ms_get_pixel(pTempBuf, width, chns, ir + 1, c, k);
				ms_add_pixel(pImageBuf, width, chns, r, c, k, val);
			}
		}
	}

	delete[] pTempBuf;

	if (crop) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

	if (0) {
		ms_dump_arr_feat2(0, "direct load", height * width * chns, height, width, chns, pImageBuf);
	}
}

string TpUtils::get_timestamp(time_t tm) {
	struct tm timeinfo;
	char buffer[80];

	if (localtime_s(&timeinfo, &tm)) TP_THROW(VERR_UNDEFINED);


	strftime(buffer, 80, "%D %T", &timeinfo);
	return string(buffer);
}

string TpUtils::get_date_8(time_t tm) {
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

string TpUtils::get_time_6(time_t tm) {
	struct tm timeinfo;
	char buffer[80];

#ifdef KA_WINDOWS
	localtime_s(&timeinfo, &tm);
	struct tm* now = &timeinfo;
#else
	struct tm* now = localtime(&tm);
#endif

	snprintf(buffer, 80, "%02d%02d%02d", now->tm_hour, now->tm_min, now->tm_sec);

	return string(buffer);
}

void TpUtils::dumpStrList(string sTitle, VStrList slList) {
	print("%s: %d elements", sTitle.c_str(), (int)slList.size());
	for (int n = 0; n < (int)slList.size(); n++) {
		print("  [%d] %s", n, slList[n].c_str());
	}
}

string TpUtils::tolower(string str) {
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	return str;
}

VValue TpUtils::seekDict(VDict dict, string sKey, VValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) return kDefaultValue;
	else return dict[sKey];
}

VDict TpUtils::mergeDict(VDict dict1, VDict dict2) {
	VDict dict;

	for (auto& it : dict1) {
		dict[it.first] = it.second;
	}

	for (auto& it : dict2) {
		if (dict1.find(it.first) != dict1.end()) continue;
		dict[it.first] = it.second;
	}

	return dict;
}

int TpUtils::find(VList list, VValue element) {
	int nth = 0;

	for (auto& val : list) {
		if (val == element) return nth;
		nth++;
	}

	return -1;
}

cv::Mat TpUtils::drawImageGrid(ETensorList tensors, string image_shape) {
	VShape xshape = tensors[0].shape();

	const char* porder = image_shape.c_str();

	int npos = (int)(strchr(porder, 'N') - porder);
	int hpos = (int)(strchr(porder, 'H') - porder);
	int wpos = (int)(strchr(porder, 'W') - porder);
	int cpos = (int)(strchr(porder, 'C') - porder);

	int H = (int)xshape[hpos];
	int W = (int)xshape[wpos];
	int C = (int)xshape[cpos];

	int show_size = (int)xshape[0];
	int show_kind = (int)tensors.size();

	int color_type = (C == 1) ? CV_8UC1 : CV_8UC3;

	cv::Scalar initData = (C == 1) ? cv::Scalar(0) : cv::Scalar(0, 0, 0);

	cv::Mat mat((H + 5) * show_kind + 5, (W + 5) * show_size + 5, color_type, initData);

	VList pos{ 0, 0, 0, 0 };

	for (int n = 0; n < show_size; n++) {
		pos[npos] = n;
		for (int w = 0; w < W; w++) {
			pos[wpos] = w;
			for (int h = 0; h < H; h++) {
				pos[hpos] = h;
				for (int c = 0; c < C; c++) {
					pos[cpos] = c;
					for (int k = 0; k < show_kind; k++) {
						float fpixel = tensors[k].getElement(pos);
						fpixel = MAX(MIN(fpixel, 1.0f), -1.0f);
						unsigned char cpixel = (unsigned char)((fpixel + 1.0f) * 127.5f);
						if (C == 1) mat.at<uchar>((H + 5) * k + h + 5, n * (W + 5) + w + 5) = cpixel;
						else mat.at<cv::Vec3b>((H + 5) * k + h + 5, n * (W + 5) + w + 5)[c] = cpixel;
					}
				}
			}
		}
	}

	return mat;
}

cv::Mat TpUtils::createCanvas(int height, int width, int chns) {
	int color_type = (chns == 1) ? CV_8UC1 : CV_8UC3;
	cv::Scalar initData = (chns == 1) ? cv::Scalar(0) : cv::Scalar(0, 0, 0);

	cv::Mat mat(height, width, color_type, initData);

	return mat;
}

void TpUtils::drawTensorImage(cv::Mat mat, ETensor tensor, cv::Rect rect, int dataIdx) {
	VShape xshape = tensor.shape();

	int N = (int)xshape[0];
	int H = (int)xshape[1];
	int W = (int)xshape[2];
	int C = (int)xshape[3];

	if (dataIdx < 0 || dataIdx >= N) TP_THROW(VERR_OUT_OF_RANGE);
	if (rect.height % H != 0) TP_THROW(VERR_IMAGE_HEIGHT);
	if (rect.width % W != 0) TP_THROW(VERR_IMAGE_WIDTH);
	if (C != 3) TP_THROW(VERR_IMAGE_CHANNEL);
	
	int hratio = rect.height / H;
	int wratio = rect.width / W;

	for (int w = 0; w < rect.width; w++) {
		for (int h = 0; h < rect.height; h++) {
			for (int c = 0; c < C; c++) {
				VList pos{ dataIdx, h / hratio, w / wratio, c };

				float fpixel = tensor.getElement(pos);
				fpixel = MAX(MIN(fpixel, 1.0f), -1.0f);
				unsigned char cpixel = (unsigned char)((fpixel + 1.0f) * 127.5f);
				if (C == 1) mat.at<uchar>(rect.y + h, rect.x + w) = cpixel;
				else mat.at<cv::Vec3b>(rect.y + h, rect.x + w)[c] = cpixel;
			}
		}
	}
}

void TpUtils::addMapImage(cv::Mat mat, ETensor tensor, cv::Rect rect, int ndat, int nchn, int color_flags) {
	VShape xshape = tensor.shape();

	int N = (int)xshape[0];
	int H = (int)xshape[1];
	int W = (int)xshape[2];
	int C = (int)xshape[3];

	if (ndat < 0 || ndat >= N) TP_THROW(VERR_OUT_OF_RANGE);
	if (nchn < 0 || nchn >= C) TP_THROW(VERR_OUT_OF_RANGE);

	if (rect.height % H != 0) TP_THROW(VERR_IMAGE_HEIGHT);
	if (rect.width % W != 0) TP_THROW(VERR_IMAGE_WIDTH);

	int hratio = rect.height / H;
	int wratio = rect.width / W;

	for (int c = 0; c < 3; c++) {
		if (((color_flags >> c) & 0x01) == 0) continue;

		for (int w = 0; w < rect.width; w++) {
			for (int h = 0; h < rect.height; h++) {
				VList pos{ ndat, h / hratio, w / wratio, nchn };

				float fpixel = tensor.getElement(pos);
				fpixel = MAX(MIN(fpixel, 1.0f), -1.0f);
				unsigned char cpixel = (unsigned char)((fpixel + 1.0f) * 127.5f);
				mat.at<cv::Vec3b>(rect.y + h, rect.x + w)[c] = cpixel;
			}
		}
	}
}

void TpUtils::addFilterImage(cv::Mat mat, ETensor tensor, cv::Rect rect, int xchn, int ychn, int color_flags, float coef) {
	VShape kshape = tensor.shape();

	int H = (int)kshape[0];
	int W = (int)kshape[1];
	int X = (int)kshape[2];
	int Y = (int)kshape[3];

	if (xchn < 0 || xchn >= X) TP_THROW(VERR_OUT_OF_RANGE);
	if (ychn < 0 || ychn >= Y) TP_THROW(VERR_OUT_OF_RANGE);

	int hratio = rect.height / H;
	int wratio = rect.width / W;

	int hgap = (rect.height - H * hratio) / 2;
	int wgap = (rect.width - W * wratio) / 2;

	for (int c = 0; c < 3; c++) {
		if (((color_flags >> c) & 0x01) == 0) continue;

		for (int w = 0; w < W * wratio; w++) {
			for (int h = 0; h < H * hratio; h++) {
				VList pos{ h / hratio, w / wratio, xchn, ychn };

				float fpixel = (float)tensor.getElement(pos) * coef;
				fpixel = MAX(MIN(fpixel, 1.0f), -1.0f);
				unsigned char cpixel = (unsigned char)((fpixel + 1.0f) * 127.5f);
				mat.at<cv::Vec3b>(rect.y + h + hgap, rect.x + w + wgap)[c] = cpixel;
			}
		}
	}
}

void TpUtils::copyCanvas(cv::Mat dstMat, cv::Mat srcMat) {
	//TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

void TpUtils::displayImage(string name, cv::Mat mat, int milliSec) {
	cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
	cv::imshow(name, mat);
	cv::waitKey(milliSec);
	cv::destroyAllWindows();
}

void TpUtils::saveImage(string path, string name, cv::Mat mat) {
	try {
		string filename = path + name + TpUtils::get_time_6(time(NULL)) + ".jpg";
		cv::imwrite(filename, mat);
	}
	catch (...) {
		TP_THROW(VERR_FILE_WRITE);
	}
}

string TpUtils::toJsonString(VDict dict) {
	return dict.desc();
}

string TpUtils::toJsonString(VList list) {
	return list.desc();
}

string TpUtils::to_string(VDataType type) {
	switch (type) {
	case VDataType::float32: return "float32";
	case VDataType::int32:   return "int32";
	case VDataType::int64:   return "int64";
	case VDataType::uint8:   return "uint8";
	case VDataType::bool8:   return "bool8";
	default:
		TP_THROW(VERR_CONDITIONAL_STATEMENT);
	}
}

VDataType TpUtils::to_data_type(string type) {
	if (type == "float32") return VDataType::float32;
	else if (type == "int32") return VDataType::int32;
	else if (type == "int64") return VDataType::int64;
	else if (type == "uint8") return VDataType::uint8;
	else if (type == "bool8") return VDataType::bool8;
	else TP_THROW(VERR_CONDITIONAL_STATEMENT);
}

int TpUtils::byte_size(VDataType type) {
	switch (type) {
	case VDataType::float32:
	case VDataType::int32:
		return 4;

	case VDataType::int64:
		return 8;

	case VDataType::uint8:
	case VDataType::bool8:
		return 1;

	default:
		TP_THROW(VERR_CONDITIONAL_STATEMENT);
	}
}

VDict TpUtils::TensorDictToDict(ETensorDict dict, bool upload) {
	VDict handles;

	for (auto& it : dict) {
		if (upload) it.second.upload();
		handles[it.first] = (VHTensor)it.second;
	}

	return handles;
}

VList TpUtils::TensorListToList(ETensorList dict, bool upload) {
	VList handles;

	for (auto& it : dict) {
		if (upload) it.upload();
		handles.push_back((VHTensor)it);
	}

	return handles;
}

VDict TpUtils::LossDictToDict(ELossDict dict) {
	VDict handles;

	for (auto& it : dict) {
		handles[it.first] = (VHLoss)it.second;
	}

	return handles;
}

ETensorDict TpUtils::DictToTensorDict(ENN nn, VDict handles) {
	ETensorDict tensors;

	for (auto& it : handles) {
		tensors[it.first] = ETensor(nn, (VHTensor)it.second, true, false);
	}

	return tensors;
}

VStrList TpUtils::ListToStrList(VList list) {
	VStrList slist;

	for (auto& it : list) {
		slist.push_back((string)it);
	}

	return slist;
}

ETensorList TpUtils::ListToTensorList(ENN nn, VList handles, bool download) {
	ETensorList tensors;

	for (auto& it : handles) {
		nn.registTensorHandle((VHTensor)it);
		ETensor tensor(nn, (VHTensor)it, true, false);
		if (download) tensor.downloadData();
		tensors.push_back(tensor);
	}

	return tensors;
}

ETensorDicts TpUtils::DictToTensorDicts(ENN nn, VDict handles) {
	ETensorDicts tensors;

	for (auto& it : handles) {
		tensors[it.first] = DictToTensorDict(nn, (VDict)it.second);
	}

	return tensors;
}

EModuleDict TpUtils::DictToModuleDict(ENN nn, VDict handles) {
	EModuleDict modules;

	for (auto& it : handles) {
		modules[it.first] = EModule(nn, (VHModule)it.second);
	}

	return modules;
}

EModuleList TpUtils::ListToModuleList(ENN nn, VList handles) {
	EModuleList modules;

	for (auto& it : handles) {
		modules.push_back(EModule(nn, (VHModule)it));
	}

	return modules;
}

string TpUtils::id_to_token(int64 id) {
	return std::to_string(id);
}

int64 TpUtils::token_to_id(string token) {
	return _atoi64(token.c_str());
}

VValue TpUtils::deep_copy(VValue value) {
	switch (value.type()) {
	case VValueType::list:
	{
		VList srcList = value;
		VList newList;

		for (auto& it : srcList) newList.push_back(deep_copy(it));

		return newList;
	}
	case VValueType::dict:
	{
		VDict srcDict = value;
		VDict newDict;

		for (auto& it : srcDict) newDict[it.first] = deep_copy(it.second);

		return newDict;
	}
	case VValueType::shape:
	{	VShape temp = value;
		return temp.copy();
	}
	default:
		return value;
	}
}

void TpUtils::FileDump(string filepath) {
#ifdef KA_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#endif

	FILE* fin = fopen(filepath.c_str(), "rb");

	int nth = 0;

	char bytes[49];
	char chars[17];

	while (!feof(fin)) {
		unsigned char ch = fgetc(fin);

		sprintf(bytes + (nth % 16) * 3, "%02X ", ch);
		sprintf(chars + (nth % 16), "%c", (ch >= 0x20 && ch < 0x7f) ? ch : '.');

		if (++nth % 16 == 0) {
			print("%04X(%4d)    %-50s %s", nth - 16, nth - 16, bytes, chars);
		}
	}

	if (nth % 16) {
		print("%04X(%4d)    %-50s %s", nth / 16 * 16, nth / 16 * 16, bytes, chars);
	}

	fclose(fin);
}

ETensor TpUtils::ms_loadNumpyArray(ENN nn, FILE* fin) {
	if (fgetc(fin) != 'N') TP_THROW(VERR_UNDEFINED);
	if (fgetc(fin) != 'U') TP_THROW(VERR_UNDEFINED);
	if (fgetc(fin) != 'M') TP_THROW(VERR_UNDEFINED);
	if (fgetc(fin) != 'P') TP_THROW(VERR_UNDEFINED);
	if (fgetc(fin) != 'Y') TP_THROW(VERR_UNDEFINED);

	if (fgetc(fin) != 0x01) TP_THROW(VERR_UNDEFINED);
	if (fgetc(fin) != 0x00) TP_THROW(VERR_UNDEFINED);
	if (fgetc(fin) != 0x76) TP_THROW(VERR_UNDEFINED);
	if (fgetc(fin) != 0x00) TP_THROW(VERR_UNDEFINED);

	VDict header = JsonParser::ParseFile(fin);

	if ((bool)header["fortran_order"]) TP_THROW(VERR_UNDEFINED);
	
	VShape shape = header["shape"];
	string descr = header["descr"];

	while (fgetc(fin) != 0x0A);

	ETensor data;

	if (descr == "<f4") {
		int test1 = ftell(fin);
		data = ETensor(nn, shape, VDataType::float32, VDataType::float32, fin);
		int test2 = ftell(fin);
		int nnn = 0;
	}
	else if (descr == "<f8") {
		data = ETensor(nn, shape, VDataType::float32, VDataType::float64, fin);
	}
	else if (descr == "<i8") {
		data = ETensor(nn, shape, VDataType::int64, VDataType::int64, fin);
	}
	else TP_THROW(VERR_UNDEFINED);

	return data;
}

ETensorDict TpUtils::LoadNPZ(ENN nn, string filepath) {
#ifdef KA_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#endif

	FILE* fin = fopen(filepath.c_str(), "rb");

	ETensorDict tensors;

	int nth = 0;

	while (true) {
		unsigned char ch = fgetc(fin);
		
		if (ch == 0xFF) break;

		if (ch == 0x50) {
			for (int n = 0; n < 29; n++) fgetc(fin);
			string name;
			while (true) {
				ch = getc(fin);
				if (ch == 0xFF || ch == 0x01) break;
				if (ch >= 0x20 && ch <= 0x7F) name += (char)ch;
			}
			if (name == "") {
				break;
			}
			for (int n = 0; n < 20; n++) ch = fgetc(fin);
			if (ch == 0x93) {
				int test1 = ftell(fin);
				tensors[name] = ms_loadNumpyArray(nn, fin);
				int test2 = ftell(fin);
				int nnn = 0;
			}
			else {
				TP_THROW(VERR_UNDEFINED);
			}
		}
		else if (ch == 0x93) {
			string name;

			while (true) {
				name = "array_" + std::to_string(nth++);
				if (tensors.find(name) == tensors.end()) break;
			}

			tensors[name] = ms_loadNumpyArray(nn, fin);
		}
	}

	fclose(fin);
	
	return tensors;
}
