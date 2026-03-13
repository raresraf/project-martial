/**
 * @file 50A.CPP
 * @brief Solves the "Domino Piling" problem.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * 1. Read the board dimensions M (as 'n') and N (as 'n1').
 * 2. Calculate the total area of the board: area = M * N.
 * 3. Each domino has an area of 2. The maximum number of non-overlapping
 *    dominoes is the total area divided by 2, using integer division.
 * 4. Print the result.
 *
 * Time Complexity: O(1), as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int n;  // First dimension of the board.
  int n1; // Second dimension of the board.
  
  // Read the dimensions from standard input.
  cin >> n >> n1;
  
  int s; // Variable to store the total area of the board.
  s = n * n1;
  
  int div; // Variable to store the result.
  // Integer division by 2 gives the maximum number of 2x1 dominoes.
  div = s / 2;
  
  // Print the final calculated number of dominoes.
  cout << div;
  
  return 0;
}
