#pragma once

#include "../utils/tp_common.h"

// Representational State Transfer
/////////////////////////////////////////////////////////////////////////////////////////
/*
Open Port
WIndows 10 settings -> Network & Internet -> Windows Firewall -> Advanced Settings
Inbound, Outbound Rule -> add port 9090 ( You can chose port number whatever. )
*/
/////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstring>

#include <cpprest/http_listener.h>
#include <cpprest/http_client.h>

class ApiConn;
class DataBaseConn;

enum class http_method { GET, POST, PUT, PUSH, DELETE };

void RestfulDebug(char* FunctionName, char* FileName, int FileLine);

#define TP_SERVER_CALL(fcall) fcall

struct RestTransactionItem {
	RestTransactionItem() { m_pData = NULL; }
	~RestTransactionItem() { free(m_pData); }

	http_method m_method;
	string m_hostaddr;

	VList m_paths;
	VDict m_queries;

	int m_statusCode;
	VDict m_response;

	void* m_pData;
	int64 m_nByteSize;
};

class NNRestServerBase {
public:
	NNRestServerBase();
	virtual ~NNRestServerBase();

	void open(string addr);
	void openService();

protected:
	web::http::experimental::listener::http_listener* m_pServer; // Listener = Server

	bool m_serverAsyncGET(void);
	bool m_serverAsyncPUT(void);
	bool m_serverAsyncPOST(void);
	bool m_serverAsyncDELETE(void);

	virtual void m_processRequest(RestTransactionItem* pItem);
};

class NNRestCbServer : public NNRestServerBase {
public:
	NNRestCbServer();
	virtual ~NNRestCbServer();

	void open(string addr);
	void setClient(ApiConn* pApiConn, string callbackToken);

protected:
	map<string, string> m_tokenMap;

	void m_processRequest(RestTransactionItem* pItem);

	void m_processEngineFamily(RestTransactionItem* pItem);
	void m_processFunctionFamily(RestTransactionItem* pItem);
};

/*
class NNRestServer : public NNRestServerBase {
public:
	NNRestServer();
	virtual ~NNRestServer();

	void open(string http_server_addr, string db_server_addr, string username, string password, string database, string pipename, int port);
	void installKaiDB(string root_password, string root_email);

protected:
	DataBaseConn* m_pDbConn;

	static mutex ms_refcntMutex;

	void m_processRequest(RestTransactionItem* pItem);

	void m_processConnFamily(RestTransactionItem* pItem);
	void m_processUserFamily(RestTransactionItem* pItem);
	void m_processRoleFamily(RestTransactionItem* pItem);
	void m_processModelFamily(RestTransactionItem* pItem);
	void m_processEngineFamily(RestTransactionItem* pItem);
	void m_processInternalFamily(RestTransactionItem* pItem);

	void m_saveLayerInfo(int mid, int parentid, VDict info, int weight=0);
	VDict m_loadLayerInfo(int mid);
	VList m_loadChildrenLayerInfo(int parentid);

	//void m_fetchChildrenLayerToEngine(int lid, int parentid, VHModule hModule, int64 sid, VHSession hSession);
	//VHModule m_fetchLayerToEngine(VDict rec, int64 sid, VHSession hSession);

protected:
	void m_registSessionHandle(int64 sid, VHandle hHandle);
	void m_eraseSessionHandle(int64 sid, VHandle hHandle);

	static void ms_modelCbForwardHandler(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbTensorBuf, const VExBuf** ppResultBuf);
	static void ms_modelCbBackwardHandler(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbTensorBuf, const VExBuf* pCbGradBuf, const VExBuf** ppResultBuf);
	static void ms_modelCbClose(VHSession hSession, const VExBuf* pResultBuf);

	static void ms_funcCbForwardHandler(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, const VExBuf** ppResultBuf);
	static void ms_funcCbBackwardHandler(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pGradListBuf, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, int nth, const VExBuf** ppResultBuf);
	static void ms_funcCbClose(VHSession hSession, const VExBuf* pResultBuf);

	static VDict ms_execCallbackTransaction(string url, string mainMenu, string subMenu, VDict requestArgs);
};
*/