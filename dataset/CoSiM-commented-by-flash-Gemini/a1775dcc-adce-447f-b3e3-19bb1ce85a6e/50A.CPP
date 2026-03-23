/**
 * @file 50A.CPP
 * @brief This competitive programming solution addresses the "Domino Piling" problem.
 *        The goal is to determine the maximum number of 2x1 dominoes that can be
 *        placed on an M x N rectangular chessboard.
 * Algorithm: Direct Mathematical Calculation.
 *            Each domino covers exactly two squares. Therefore, the maximum number
 *            of dominoes that can be placed on an M x N board is half of the total
 *            area of the board, as long as each domino covers two squares. Since
 *            dominoes cannot overlap, and cannot extend beyond the board, the calculation
 *            (M * N) / 2 directly yields the maximum number of dominoes.
 * Time Complexity: O(1), as it involves a constant number of arithmetic operations and input/output operations.
 * Space Complexity: O(1), as only a few fixed-size integer variables are used.
 */
/*
Codeforces Beta Round #47 - A. Domino piling
https://codeforces.com/problemset/problem/50/A
*/

#include <bits/stdc++.h> // Includes various standard library headers, including iostream.
using namespace std;

// Macro for fast I/O: unties C++ streams from C std streams and unties cin from cout.
#define FAST_INP                                                               
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL)

/**
 * @brief Main function of the program. Calculates and prints the maximum number of dominoes.
 * Functional Utility: Serves as the entry point for the "Domino Piling" problem.
 */
int main() {
  FAST_INP; // Optimizes C++ standard streams for faster input/output.
  int m, n; // Declares integers to store the dimensions (rows and columns) of the chessboard.
  cin >> m >> n; // Reads the two dimensions from standard input.
  
  // Calculates the total area (m * n) and divides by 2 to get the maximum number of dominoes.
  // The integer division automatically handles cases where the area is odd (one square left uncovered).
  cout << (m * n) / 2 << "
"; // Prints the result followed by a newline.
  return 0;                    // Indicates successful program termination.
}
