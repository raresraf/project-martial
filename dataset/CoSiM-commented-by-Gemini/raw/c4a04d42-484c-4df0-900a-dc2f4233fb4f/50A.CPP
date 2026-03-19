/**
 * @file 50A.CPP
 * @brief Optimal solution for the "Domino Piling" problem (Codeforces 50A).
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * Algorithm: Area Division
 * The problem is solved with a simple and direct mathematical formula. The total
 * area of the board is M * N. Since each domino has an area of 2, the maximum
 * number of dominoes that can fit is the total area divided by 2, using
 * integer division. This is the most optimal solution.
 *
 * Time Complexity: O(1), as it involves a few basic arithmetic operations.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // n, m: the dimensions of the rectangular board.
  int n, m;
  // dominoes: the result, storing the maximum number of dominoes.
  int dominoes;

  cin >> n >> m;

  // Core Logic: Calculate the maximum number of dominoes by dividing the total
  // area of the board (n * m) by the area of a single domino (2).
  // Integer division automatically handles the floor value.
  dominoes = (n * m) / 2;
  cout << dominoes << endl;
}

// g++ -std=c++11 -O2 -Wall 50A.cpp -o out