/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem (Codeforces 50A).
 *
 * This program calculates the maximum number of 1x2 dominoes that can be
 * placed on an M x N rectangular board, where each domino covers exactly
 * two squares.
 *
 * Algorithm: Mathematical calculation. The maximum number of dominoes is
 * simply half the total area of the board.
 * Time Complexity: O(1) as it involves a few arithmetic operations.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Main function to calculate the maximum number of dominoes.
 *
 * Reads the dimensions of the board (M and N) from standard input
 * and prints the maximum number of 1x2 dominoes that can be placed on it.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare integers 'm' and 'n' for the dimensions of the board.
  int n, m;
  // Read the dimensions 'm' and 'n' from standard input.
  cin >> m >> n;
  // Calculate and print the maximum number of dominoes, which is (m * n) / 2.
  cout << (m * n) / 2;
  return 0;
}
