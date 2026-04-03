/**
 * @file 50A.cpp
 * @brief Solution to the "Domino Piling" problem.
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * Algorithm:
 * 1. Read the dimensions of the board, M and N.
 * 2. Calculate the total area of the board (M * N).
 * 3. The maximum number of dominoes is the total area divided by the area of one domino (2).
 *    This is equivalent to integer division of (M * N) by 2.
 * 4. Print the result.
 *
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
#include <iostream>
using namespace std;

// m, n: dimensions of the board, a: result (number of dominoes).
int m, n, a;

int main() {
  // Read the dimensions of the board.
  cin >> m >> n;
  
  // The logic in the original code is slightly more complex than needed.
  // The number of dominoes is simply the integer division of the area by 2.
  // if ((m * n) % 2 == 0) {
  //   a = (m * n) / 2;
  // } else {
  //   a = ((m * n) - 1) / 2;
  // }
  // A simpler way is:
  a = (m * n) / 2;
  
  // Print the maximum number of dominoes.
  cout << a;
}