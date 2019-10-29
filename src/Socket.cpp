#include "Socket.h"

Socket::Socket()
{
    type = TCPConfig::TCPtype;
    protocol = TCPConfig::TCPprotocol;
    socket_address.sin_family = TCPConfig::TCPfamily;
    socket_address.sin_port = htons(TCPConfig::TCPport);
    inet_pton(socket_address.sin_family, TCPConfig::TCPip, &(socket_address.sin_addr));
}


Socket::~Socket()
{
    close();
}

int Socket::open()
{
    char msg[] = "Client:transmitter";
    sockfd = socket(TCPConfig::TCPfamily, TCPConfig::TCPtype, TCPConfig::TCPprotocol);
    if(sockfd < 0)
    {
        printf("Could not establishe connection to %s on port %d\n", TCPConfig::TCPip, TCPConfig::TCPport);
        return 0;
    }
    if(connect(sockfd, (struct sockaddr*)&socket_address, sizeof(socket_address)) < 0)
    {
        printf("Could not connect to server\n");
        return 0;
    }
    printf("Connecting to TCP server\n");
    active = true;
    send(msg, sizeof(msg)-1); // "-1" is added to prevent sending the termination charachter '\0'
    wait();
    return 1;
}
template <typename T>
void Socket::send(T* ptr, int count)
{
    printf("Writing to server... \n");
    int result = 0;
    int _size = count;
    do
    {
        result = write(sockfd, (void*)ptr, _size);
        if(result == -1)
        {
            printf("Error sending to server: %d\n",errno);
            _size = 0;
        }
        else
        {
            //printf("Send %ld Bytes, result %ld \n", _size, result);
            _size -= result;
        }
    }
    while(_size > 0);
}
template void Socket::send<tcp_header>(tcp_header*, int count);
template void Socket::send<int>(int*, int count);
template void Socket::send<long unsigned>(long unsigned*, int count);
template void Socket::send<float>(float*, int count);
template void Socket::send<char>(char*, int count);
template void Socket::send<unsigned char>(unsigned char*, int count);

void Socket::wait()
{
    int bytes = 0;
    char msg[2];
    do {
        bytes += read(sockfd, msg, 2);
    } while(msg == "OK");
    printf("Server send OK\n");
    //return;
}
void Socket::close()
{
    printf("Closing connection to server\n");
    shutdown(sockfd, 2);
}

void Socket::printHeader(tcp_header header)
{
    printf("_________\nHeader\n__________\n");
    printf("total_size: %u\n",ntohl(header.total_size));
    printf("rec_records: %u\n",ntohl(header.rec_records));
    printf("nof_channels: %u\n",ntohl(header.nof_channels));
    printf("img_height: %u\n",ntohl(header.img_height));
    printf("img_width: %u\n",ntohl(header.img_width));
    printf("format: %u\n",ntohl(header.format));
}
