/**
 * @file 50A.CPP
 * @brief Solution for Codeforces problem 50A, "Domino piling".
 *
 * @description
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @algorithm
 * The problem is solved by calculating the total area of the board and
 * dividing it by the area of a single domino.
 * 1. Read the board dimensions M and N.
 * 2. Calculate the total area (M * N).
 * 3. The maximum number of dominoes is `(M * N) / 2`. Integer division
 *    correctly handles boards with both even and odd total areas.
 *
 * @time_complexity O(1).
 * @space_complexity O(1).
 */
//
// Created by Taewoo Kim on 2/7/2017.
//
#include <iostream>
using namespace std;

int main() {
  // m and n are the dimensions of the rectangular board.
  int m, n;
  cin >> m >> n;
  
  // The maximum number of dominoes is the total board area divided by 2.
  cout << (m * n) / 2;
  return 0;
}
