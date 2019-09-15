#ifndef SOCKET_H_
#define SOCKET_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "TCPConfig.h"

using namespace std;

typedef struct{
    unsigned total_size;
    unsigned rec_records;
    unsigned nof_channels;
    unsigned img_height;
    unsigned img_width;
    unsigned color_depth;
}tcp_header;

class Socket : public TCPConfig
{
public:
    Socket();
    ~Socket();
    int open();
    void close();
    template <typename T>void writeToServer(T* ptr, size_t count);
    bool isActive(){return active;}

private:
    int sockfd = 0;
    bool active = false;
    int type;
    int protocol;
    struct sockaddr_in socket_address;
    struct hostent *server;
};

#endif
