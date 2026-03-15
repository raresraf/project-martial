/**
 * @file 231A.CPP
 * @brief Solution to a simple counting problem.
 * @details This program reads a number of sets, each containing three binary values (opinions). It counts how many of these sets have at least two of the three values set to 1. This is a common pattern in introductory competitive programming problems where participants need to count items meeting a certain threshold.
 *
 * @algorithm The algorithm is a straightforward linear scan. It iterates through each set of opinions and uses a conditional block to check if the sum of opinions is two or greater.
 * @complexity
 * - Time: O(N), where N is the number of sets, as we process each set once.
 * - Space: O(1), as the memory usage is constant regardless of the input size.
 */

#include <iostream>

using namespace std;

int main() {
  // n: The total number of problems (or sets of opinions) to process.
  // x, y, z: The binary opinions of three individuals for a single problem (1 for yes, 0 for no).
  // s: A counter to store the number of problems that will be implemented (i.e., have at least two 'yes' opinions).
  int n = 0, x = 0, y = 0, z = 0, s = 0;
  
  // Read the total number of problems.
  cin >> n;

  /**
   * @brief Main processing loop.
   * @invariant At the beginning of each iteration `i`, `s` holds the count of problems with at least two 'yes' opinions from the first `i` problems.
   */
  for (int i = 0; i < n; i++) {
    // Read the three opinions for the current problem.
    cin >> x >> y >> z;

    /**
     * @brief Conditional check for implementation.
     * This block increments the solution counter 's' if at least two of the three
     * individuals agree on the solution (i.e., their opinion is 1). The logic
     * explicitly checks all pairs. An equivalent and more concise check would be `if (x + y + z >= 2)`.
     */
    if (x == 1 && y == 1) {
      s++;
    } else if (x == 1 && z == 1) {
      s++;
    } else if (y == 1 && z == 1) {
      s++;
    }
  }

  // Output the total count of problems that will be implemented.
  cout << s << endl;
  
  return 0;
}