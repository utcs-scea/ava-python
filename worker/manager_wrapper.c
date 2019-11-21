#include <fcntl.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/ipc.h>
#include <sys/mman.h>

#include "common/cmd_channel_impl.h"
#include "common/cmd_handler.h"
#include "common/guest_mem.h"
#include "common/register.h"
#include "common/socket.h"

/* Python wrapper */
int init_manager_vsock()
{
    int listen_fd;
    struct sockaddr_vm sa_listen;
    listen_fd = init_vm_socket(&sa_listen, VMADDR_CID_ANY, WORKER_MANAGER_PORT);
    listen_vm_socket(listen_fd, &sa_listen);
    return listen_fd;
}

struct command_base *poll_client(int listen_fd, int *client_fd, int *guest_cid)
{
    struct command_base *msg = (struct command_base *)malloc(sizeof(struct command_base));
    *client_fd = accept_vm_socket(listen_fd, guest_cid);
    recv_socket(*client_fd, msg, sizeof(struct command_base));
    return msg;
}

void respond_client(int client_fd, int worker_id)
{
    struct command_base response;
    uintptr_t *worker_port;

    response.api_id = INTERNAL_API;
    worker_port = (uintptr_t *)response.reserved_area;
    *worker_port = worker_id + WORKER_PORT_BASE;
    send_socket(client_fd, &response, sizeof(struct command_base));
    close(client_fd);
}

void close_client(int client_fd)
{
    close(client_fd);
}
