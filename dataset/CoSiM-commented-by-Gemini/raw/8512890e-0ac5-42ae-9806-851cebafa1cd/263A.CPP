/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem.
 *
 * @details
 * This program calculates the minimum number of moves required to move the single '1'
 * in a 5x5 matrix to the center at (2, 2) (0-indexed). The number of moves is
 * equivalent to the Manhattan distance from the '1's current position to the center.
 *
 * Algorithm:
 * 1. Read the 5x5 matrix and find the coordinates (posx, posy) of the '1' in a single pass.
 * 2. Calculate the Manhattan distance to the center (2, 2).
 * 3. The distance is calculated using an unusual method: `sqrt((posx - 2)^2) + sqrt((posy - 2)^2)`,
 *    which is mathematically equivalent to `abs(posx - 2) + abs(posy - 2)`.
 * 4. Print the resulting distance.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5, making the loop run a constant number of times.
 * Space Complexity: O(1), for the fixed-size 5x5 array.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  int a[5][5];
  int posx, posy;
  /**
   * @brief Reads the 5x5 matrix and finds the coordinates of the '1'.
   * Pre-condition: Standard input contains 25 integers (24 zeros, one 1).
   * Post-condition: `a` is populated, and `posx`, `posy` hold the 0-indexed coordinates of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; ++j) {
      cin >> a[i][j];
      if (a[i][j] > 0) { // Equivalent to checking if the element is 1.
        posx = i;
        posy = j;
      }
    }
  }

  // The following section calculates the Manhattan distance using squares and square roots,
  // which is an elaborate way of finding the absolute value of the differences.

  // Calculate the squared difference in rows from the center.
  int answer_posx = pow(posx - 2, 2);
  // Calculate the squared difference in columns from the center.
  int answer_posy = pow(posy - 2, 2);

  // Taking the square root of the square is equivalent to taking the absolute value.
  answer_posx = sqrt(answer_posx);
  answer_posy = sqrt(answer_posy);

  // The total number of moves is the sum of the row and column moves (Manhattan distance).
  int answer = answer_posx + answer_posy;
  cout << answer;
}