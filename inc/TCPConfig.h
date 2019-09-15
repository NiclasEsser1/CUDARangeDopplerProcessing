#ifndef TCPCONFIG_H_
#define TCPCONFIG_H_

#include <sys/types.h>
#include <sys/socket.h>

class TCPConfig
{

public:
    const int TCPtype = SOCK_STREAM;
    const short TCPfamily = AF_INET;
    const unsigned short TCPport = 1337;
    const char* TCPip = "127.0.0.1";
    const int TCPprotocol = 0;
};

#endif
