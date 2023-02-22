#pragma once

#include "../utils/tp_common.h"
#include "../connect/tp_nn_server.h"

class ApiConn;

class NNRestClientBase {
public:
	NNRestClientBase(string server_url);
	virtual ~NNRestClientBase();

	virtual VDict execTransaction(string mainMenu, string subMenu, VDict requestArgs, void* pData = NULL, int64 nByteSize = 0, bool bUpload = true);

protected:
	web::http::client::http_client* m_pClient;
};

class NNRestClient : public NNRestClientBase {
public:
	NNRestClient(string server_url, string client_url, ApiConn* pApiConn, VDict kwArgs);
	virtual ~NNRestClient();

	string getCallbackToken() { return m_accessToken; }

	VDict execEngineExec(string apiName, VDict apiArgs, void* pData = NULL, int64 nByteSize = 0, bool bUpload = true);
	VDict execTransaction(string mainMenu, string subMenu, VDict requestArgs, void* pData = NULL, int64 nByteSize = 0, bool bUpload = true);

protected:
	void m_connect(ApiConn* pApiConn, string client_url, VDict kwArgs);

protected:
	string m_accessToken;
	int m_expiresIn;
};

class NNRestCbClient : public NNRestClientBase {
public:
	NNRestCbClient(string url);
	virtual ~NNRestCbClient();

protected:
};
