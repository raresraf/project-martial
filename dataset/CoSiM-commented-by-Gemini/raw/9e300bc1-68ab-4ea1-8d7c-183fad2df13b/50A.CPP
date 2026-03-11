/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes on a rectangular board.
 *
 * This program solves the "Domino Piling" problem. Given a rectangular board
 * of M by N squares, the goal is to find the maximum number of 2x1 dominoes
 * that can be placed on the board.
 *
 * @algorithm The total number of squares on the board is M * N. Since each
 * domino covers exactly two squares, the maximum number of dominoes that can
 * be placed is the total number of squares divided by 2. Integer division
 * inherently handles both even and odd total areas correctly.
 *
 * @time_complexity O(1), as the solution involves a single arithmetic calculation.
 * @space_complexity O(1), as it only uses a couple of variables to store the dimensions.
 */
#include <iostream>
using namespace std;
int main() {
  int M, N; // Dimensions of the rectangular board.
  
  // Read the dimensions from standard input.
  cin >> M >> N;

  // Calculate and print the maximum number of 2x1 dominoes.
  // The total area is M * N. Each domino covers 2 squares.
  // Integer division (M * N / 2) gives the maximum number of dominoes.
  cout << M * N / 2;
}