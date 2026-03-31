/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces (Problem 263A).
 * This program calculates the minimum number of moves to bring the '1' in a 5x5 matrix
 * to the center position (row 2, column 2, 0-indexed).
 *
 * Algorithm: Manhattan Distance. The minimum moves correspond to the sum of absolute
 * differences in row and column indices between the '1' and the center.
 * Time Complexity: O(1) as the matrix size is fixed (5x5).
 * Space Complexity: O(1) for storing the fixed-size matrix and a few variables.
 */

#include <algorithm> // Required for std::abs (implicitly, though cmath pow/sqrt are used)
#include <cmath>     // Required for std::pow and std::sqrt
#include <iostream>  // Required for standard input/output streams (cin, cout)
#include <string>    // Not strictly needed for this problem but often included.
#include <vector>    // Not strictly needed for this problem but often included.

using namespace std;

int main() {
  /**
   * @brief Main function to solve the Beautiful Matrix problem.
   * Reads a 5x5 matrix to find the position of the '1' element, then calculates
   * and prints the Manhattan distance from that position to the center of the matrix.
   */
  int a[5][5]; // @brief 5x5 matrix to store input values.
  int posx, posy; // @brief Variables to store the row and column of the '1' element.
  
  // Block Logic: Read the 5x5 matrix and find the coordinates of the '1' element.
  // Pre-condition: The input is a 5x5 matrix with exactly one '1' and rest '0's.
  // Invariant: After the loops, 'posx' and 'posy' will hold the 0-indexed coordinates
  //            of the '1' element.
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; ++j) {
      cin >> a[i][j];
      if (a[i][j] > 0) { // Inline: Assuming '1' is the only positive number.
        posx = i;
        posy = j;
      }
    }
  }
  
  // Block Logic: Calculate the Manhattan distance to the center (2,2).
  // The center is at (2,2) for a 0-indexed 5x5 matrix.
  // The distance is |posx - 2| + |posy - 2|.
  // Using pow(x-y, 2) and sqrt is an indirect way to get abs(x-y).
  // Functional Utility: std::abs(posx - 2) would be a more direct way to calculate
  //                     the absolute difference.
  int answer_posx = pow(posx - 2, 2); // Inline: Calculates (posx - 2)^2
  int answer_posy = pow(posy - 2, 2); // Inline: Calculates (posy - 2)^2
  answer_posx = sqrt(answer_posx); // Inline: Calculates sqrt((posx - 2)^2) = |posx - 2|
  answer_posy = sqrt(answer_posy); // Inline: Calculates sqrt((posy - 2)^2) = |posy - 2|
  int answer = answer_posx + answer_posy; // Inline: Sums absolute differences to get Manhattan distance.
  
  // Block Logic: Output the calculated minimum number of moves.
  cout << answer;
}