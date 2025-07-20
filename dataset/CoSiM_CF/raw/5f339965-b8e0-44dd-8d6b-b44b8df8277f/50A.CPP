#include <bits/stdc++.h>
using namespace std;

inline bool even(int a) { return a % 2 == 0; }

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int M, N;
  cin >> M >> N;
  cout << ((even(M) || even(N)) ? M * N / 2 : (M * N - 1) / 2) << "\n";
}

/*
Say the max amount of dominoes that can totally fit in an MxN board.

1. Each domino completely covers two squares.
2. No two dominoes overlap.
3. Each domino lies entirely inside the board. It is allowed to touch the edges
of the board.

Input
M, N  1 <= M <= N <= 16

*/