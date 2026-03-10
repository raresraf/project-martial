/**
 * @file 50A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Domino piling").
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular M x N board.
 *
 * @algorithm Direct Mathematical Formula. The total area of the board is M * N.
 * Each domino covers an area of 2. The maximum number of non-overlapping
 * dominoes is the total area divided by 2. Integer division inherently handles
 * the floor operation required when the total area is odd.
 *
 * @time_complexity O(1), as it's a direct arithmetic calculation.
 * @space_complexity O(1).
 */
#include <ctype.h>
#include <fstream>
#include <iostream>
using namespace std;

int main() {
  // freopen("input.txt","r",stdin);
  int M, N;
  cin >> M >> N;

  // Calculates the maximum number of dominoes by dividing the total board area by 2.
  cout << (M * N) / 2;
  return 0;
}