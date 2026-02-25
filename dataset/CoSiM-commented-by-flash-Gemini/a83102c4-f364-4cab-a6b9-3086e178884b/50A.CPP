/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces (Problem 50A).
 *        This program calculates the maximum number of 2x1 dominoes that can
 *        be placed on an M x N rectangular chessboard.
 * Algorithm: Mathematical / Direct Calculation
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */

#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to calculate the maximum number of dominoes on an M x N board.
 * @return 0 if the program executes successfully.
 */
int main() {
  int m, n; // Declare integers for the dimensions of the chessboard.
  // Read the dimensions M and N from standard input.
  cin >> m >> n;
  int count = 0; // Declare an integer to store the maximum number of dominoes.
  
  /**
   * Block Logic: Calculates the maximum number of dominoes based on the board dimensions.
   * Pre-condition: 'm' and 'n' are positive integers representing board dimensions.
   * Post-condition: 'count' holds the maximum number of 2x1 dominoes that can be placed.
   * Note: The problem simplifies to (M * N) / 2, as each domino covers 2 squares.
   */
  if (m % 2 == 0) { // If M is an even number.
    // The number of dominoes is (M/2) * N.
    count = (m / 2) * n;
  } else { // If M is an odd number.
    // The number of dominoes is (M/2) * N (for the even part) + (N/2) (for the remaining 1xN strip).
    count = (m / 2) * n + (n / 2);
  }
  // Print the calculated maximum number of dominoes followed by a newline.
  cout << count << endl;
  return 0; // Indicate successful program termination.
}
