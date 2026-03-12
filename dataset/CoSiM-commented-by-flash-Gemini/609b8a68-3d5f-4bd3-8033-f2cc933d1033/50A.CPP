/**
 * @609b8a68-3d5f-4bd3-8033-f2cc933d1033/50A.CPP
 * @brief This program solves the "Domino piling" problem (Codeforces 50A).
 * Given a rectangular board of size M x N and dominoes of size 2 x 1,
 * it calculates the maximum number of dominoes that can be placed on the board.
 * Each domino covers exactly two squares.
 *
 * Algorithm: Direct calculation based on board area and integer division, handling parity.
 * Time Complexity: O(1) as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1) for storing variables.
 */

#include <cmath>     // Included but not strictly necessary for integer arithmetic.
#include <iostream>  // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

int main() {
  int m, n; // Declares 'm' and 'n' for the dimensions of the rectangular board.
  cin >> m >> n; // Reads the dimensions M and N from standard input.
  
  /**
   * Block Logic: Determines the maximum number of dominoes based on the parity of M and N.
   * Invariant: The correct maximum number of dominoes is printed, which is floor((M * N) / 2).
   */
  if (m % 2 == 0 && n % 2 == 0) { // If both dimensions M and N are even:
    // The total number of squares (M * N) is even.
    cout << (m * n) / 2; // Inline: Max dominoes is total squares divided by 2.
  } else if (m % 2 != 0 && n % 2 != 0) { // If both dimensions M and N are odd:
    // The total number of squares (M * N) is odd. One square will always be left uncovered.
    // The formula ((m - 1) * (n - 1)) / 2 + (m + n - 2) / 2 simplifies to (m*n - 1) / 2.
    // Integer division (m * n) / 2 automatically handles the floor for odd products.
    cout << ((m - 1) * (n - 1)) / 2 + (m + n - 2) / 2; // This is equivalent to (m * n) / 2 using integer division.
  } else { // If one dimension is even and the other is odd:
    // The total number of squares (M * N) is even.
    cout << (m * n) / 2; // Inline: Max dominoes is total squares divided by 2.
  }
  return 0; // Indicates successful program termination.
}
