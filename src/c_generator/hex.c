// Copyright (c) 2024 Ole-Christoffer Granmo
// MIT License

// generate 11x11 = gcc -O3 -march=native -funroll-loops -DBOARD_DIM=11 src/c_generator/hex.c -o src/c_generator/hex
// generate 6x6  = gcc -O3 -march=native -funroll-loops -DBOARD_DIM=6  src/c_generator/hex.c -o src/c_generator/hex
// etc.
//
// then run: ./src/c_generator/hex 1000

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BOARD_DIM
#define BOARD_DIM 9
#endif

static int neighbors[6];

struct hex_game {
    int board[(BOARD_DIM + 2) * (BOARD_DIM + 2) * 2];
    int open_positions[BOARD_DIM * BOARD_DIM];
    int number_of_open_positions;
    int connected[(BOARD_DIM + 2) * (BOARD_DIM + 2) * 2];
};

void hg_init_neighbors(void) {
    neighbors[0] = -(BOARD_DIM + 2) + 1;
    neighbors[1] = -(BOARD_DIM + 2);
    neighbors[2] = -1;
    neighbors[3] = 1;
    neighbors[4] = (BOARD_DIM + 2);
    neighbors[5] = (BOARD_DIM + 2) - 1;
}

void hg_init(struct hex_game *hg) {
    for (int i = 0; i < BOARD_DIM + 2; i++) {
        for (int j = 0; j < BOARD_DIM + 2; j++) {
            hg->board[(i * (BOARD_DIM + 2) + j) * 2] = 0;
            hg->board[(i * (BOARD_DIM + 2) + j) * 2 + 1] = 0;

            if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1)
                hg->open_positions[(i - 1) * BOARD_DIM + j - 1] = i * (BOARD_DIM + 2) + j;

            hg->connected[(i * (BOARD_DIM + 2) + j) * 2] = (i == 0);
            hg->connected[(i * (BOARD_DIM + 2) + j) * 2 + 1] = (j == 0);
        }
    }
    hg->number_of_open_positions = BOARD_DIM * BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int pos) {
    hg->connected[pos * 2 + player] = 1;
    if (player == 0 && pos / (BOARD_DIM + 2) == BOARD_DIM) return 1;
    if (player == 1 && pos % (BOARD_DIM + 2) == BOARD_DIM) return 1;

    for (int i = 0; i < 6; i++) {
        int n = pos + neighbors[i];
        if (hg->board[n * 2 + player] && !hg->connected[n * 2 + player])
            if (hg_connect(hg, player, n)) return 1;
    }
    return 0;
}

int hg_winner(struct hex_game *hg, int player, int pos) {
    for (int i = 0; i < 6; i++) {
        int n = pos + neighbors[i];
        if (hg->connected[n * 2 + player])
            return hg_connect(hg, player, pos);
    }
    return 0;
}

int hg_place_piece_random(struct hex_game *hg, int player) {
    int idx = rand() % hg->number_of_open_positions;
    int pos = hg->open_positions[idx];
    hg->board[pos * 2 + player] = 1;
    hg->open_positions[idx] = hg->open_positions[hg->number_of_open_positions - 1];
    hg->number_of_open_positions--;
    return pos;
}

int hg_full(struct hex_game *hg) { return hg->number_of_open_positions == 0; }

int main(int argc, char *argv[]) {
    int total_games = argc > 1 ? atoi(argv[1]) : 100;
    srand((unsigned int)time(NULL));
    hg_init_neighbors();

    char filename[128];
    snprintf(filename, sizeof(filename), "datasets/hex_games_%d.csv", BOARD_DIM);
    FILE *f = fopen(filename, "w");
    if (!f) { perror(filename); return 1; }

    for (int i = 0; i < BOARD_DIM; i++) {
        for (int j = 0; j < BOARD_DIM; j++)
            fprintf(f, "cell_%d_%d,", i, j);
    }
    fprintf(f, "winner\n");

    struct hex_game hg;
    for (int g = 0; g < total_games; g++) {
        hg_init(&hg);
        int player = 0, winner = -1;
        while (!hg_full(&hg)) {
            int pos = hg_place_piece_random(&hg, player);
            if (hg_winner(&hg, player, pos)) { winner = player; break; }
            player = 1 - player;
        }

        for (int i = 1; i <= BOARD_DIM; i++) {
            for (int j = 1; j <= BOARD_DIM; j++) {
                int p0 = hg.board[(i * (BOARD_DIM + 2) + j) * 2];
                int p1 = hg.board[(i * (BOARD_DIM + 2) + j) * 2 + 1];
                int val = p0 ? 1 : (p1 ? -1 : 0);
                fprintf(f, "%d,", val);
            }
        }
        fprintf(f, "%d\n", winner);
    }

    fclose(f);
    printf("Saved %d games to %s\n", total_games, filename);
    return 0;
}