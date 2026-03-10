/**
 * @file 50A.CPP
 * @brief This program calculates the maximum number of 2x1 dominoes that can be placed on an M x N chessboard.
 * * Algorithm: Direct calculation based on chessboard area.
 * Time Complexity: O(1) - constant time for arithmetic operations.
 * Space Complexity: O(1) - constant space for integer variables.
 */

#include <iostream>
using namespace std;

/**
 * @brief Main function that reads chessboard dimensions and outputs the maximum dominoes.
 * Functional Utility: Orchestrates input, calculation, and output for the domino tiling problem.
 */
int main() {
  int M, N;
  // Block Logic: Reads two integers M and N representing the dimensions of the chessboard.
  cin >> M >> N;

  // Block Logic: Calculates the maximum number of 2x1 dominoes that can be placed on an M x N board.
  // This is equivalent to half the total area of the board, as each domino covers 2 squares.
  cout << M * N / 2;
}