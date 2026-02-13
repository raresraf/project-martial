/**
 * @file 50A.CPP
 * @brief Solution for the "Domino piling" problem (Codeforces 50A).
 * @author competitive-programmer
 *
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * 1. The total area of the board is M * N.
 * 2. The area of a single domino is 2 * 1 = 2.
 * 3. The maximum number of dominoes that can fit on the board is the total
 *    area of the board divided by the area of a single domino, using integer
 *    division.
 *    Formula: floor((M * N) / 2)
 *
 * Time Complexity: O(1), as the solution involves a single arithmetic calculation.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  // n and m are the dimensions of the rectangular board.
  int n, m;
  cin >> n >> m;
  
  // The variable 's' is declared but never used in the program.
  int s = 2;
  
  /**
   * The maximum number of dominoes is calculated by finding the total area
   * of the board (m * n) and performing an integer division by the area of
   * a single domino (2).
   */
  cout << m * n / 2 << endl;
  
  return 0;
}
