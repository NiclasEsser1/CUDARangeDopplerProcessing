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

typedef struct{
    unsigned total_size;
    unsigned total_nof_records;
    unsigned rec_records;
    unsigned nof_channels;
    unsigned current_channel;
    unsigned img_height;
    unsigned img_width;
    unsigned format;
}tcp_header;

using namespace std;
class Socket : public TCPConfig
{
public:
    Socket();
    ~Socket();
    int open();
    void close();
    template <typename T>void send(T* ptr, int count);
    void wait();
    bool isActive(){return active;}
    void printHeader(tcp_header header);

private:
    int sockfd = 0;
    bool active = false;
    int type;
    int protocol;
    struct sockaddr_in socket_address;
    struct hostent *server;
};

#endif
