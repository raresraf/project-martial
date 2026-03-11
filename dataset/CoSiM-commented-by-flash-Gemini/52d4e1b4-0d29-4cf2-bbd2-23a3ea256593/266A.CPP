/**
 * @file 266A.CPP
 * @brief Solution to Codeforces Problem 266A: "Stones on the Table".
 *        The problem involves counting the minimum number of stones to remove from a row
 *        so that no two adjacent stones have the same color. This is equivalent to counting
 *        the number of times an adjacent stone has the same color.
 *
 * Algorithm: Iterative scan. The algorithm iterates through the row of stones, comparing each stone
 *            with its preceding stone. If they are of the same color, a counter is incremented.
 * Time Complexity: O(N) where N is the number of stones, as the solution involves a single pass
 *                  through the string of colors.
 * Space Complexity: O(N) to store the string of colors.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  /**
   * Functional Utility: Declaration and initialization of variables.
   * `no`: Stores the total number of stones.
   * `output`: Accumulates the count of stones to be removed.
   * `lastcolor`: Stores the color of the previously processed stone.
   * `color`: Stores the color of the currently processed stone.
   * `colors`: Stores the input string representing the colors of the stones.
   */
  int no;
  int output = 0;
  char lastcolor;
  char temp; // Unused variable, potential for removal
  char color;
  string colors;

  // Input: Read the number of stones and the string of colors.
  cin >> no;
  cin >> colors;

  /**
   * Conditional Logic: Proceed with calculation only if there is more than one stone.
   * Rationale: If there's only one stone or no stones, no adjacent stones can have the same color.
   */
  if (no > 1) {
    // Initialization for the loop: Set the first color to 'colors.at(1)' for comparison with 'colors.at(0)'.
    // Note: The assignment 'color = colors.at(1)' here is redundant as it will be reassigned inside the loop.
    color = colors.at(1);
    /**
     * Block Logic: Iterate through the stones from the second stone to the last.
     * Pre-condition: `colors` contains the stone sequence, `no` is its length.
     * Invariant: `i` represents the current stone's index, `color` is `colors[i]`,
     *            and `lastcolor` is `colors[i-1]`.
     */
    for (int i = 1; i < no; i++) {
      color = colors.at(i);
      lastcolor = colors.at(i - 1);
      /**
       * Conditional Logic: Checks if the current stone has the same color as the previous stone.
       * If true, increment `output` as this stone needs to be removed.
       */
      if (lastcolor == color) {
        output++;
      }
    }
  }
  // Output: Print the total number of stones that need to be removed.
  cout << output;
  return 0;
}
