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
    char hello[] = "From Client:    Hello Server";
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
    active = true;
    return 1;
    //send(sockfd, hello, sizeof(hello),MSG_OOB);
}
template <typename T>
void Socket::writeToServer(T* ptr, size_t count)
{
    //printf("Writing to server... size: %ld\n", sizeof(T)*count);
    size_t result = 0;
    size_t _size = count * sizeof(T);
    if(active)
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
template void Socket::writeToServer<tcp_header>(tcp_header*, size_t count);
template void Socket::writeToServer<int>(int*, size_t count);
template void Socket::writeToServer<long unsigned>(long unsigned*, size_t count);
template void Socket::writeToServer<float>(float*, size_t count);
template void Socket::writeToServer<char>(char*, size_t count);

void Socket::close()
{

}
