/**
 * @file 50A.CPP
 * @brief This program calculates the maximum number of 1x2 or 2x1 dominoes that can be
 *        placed on a rectangular board of size M x N without overlapping.
 *
 * Algorithm:
 * 1. Read the dimensions of the board, M and N.
 * 2. Calculate the total area of the board (M * N).
 * 3. Since each domino covers 2 squares, the maximum number of dominoes is
 *    the total area divided by 2 (integer division).
 * 4. Print the result.
 *
 * Time Complexity: O(1) as it involves basic arithmetic operations.
 * Space Complexity: O(1) as only a few integer variables are used.
 */

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates reading board dimensions, calculating
 * the maximum dominoes, and printing the result.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares integers 'n' and 'm' for board dimensions,
  //              and 'dominoes' to store the calculated maximum number of dominoes.
  int n, m, dominoes;

  // Block Logic: Reads the two integer dimensions (n and m) of the board from standard input.
  cin >> n >> m;

  /**
   * Block Logic: Calculates the maximum number of dominoes that can be placed.
   * Since each domino covers 2 squares, the total number of squares (n * m) is divided by 2.
   * Integer division automatically handles cases where the total number of squares is odd,
   * effectively taking the floor of the division.
   * Functional Utility: Determines the optimal packing density of dominoes on the given board.
   */
  dominoes = (n * m) / 2;
  // Block Logic: Prints the calculated maximum number of dominoes, followed by a newline.
  cout << dominoes << endl;
}

// g++ -std=c++11 -O2 -Wall 50A.cpp -o out