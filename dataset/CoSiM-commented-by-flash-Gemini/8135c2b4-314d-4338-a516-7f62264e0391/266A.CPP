/**
 * @file 266A.CPP
 * @brief Solution to Codeforces problem 266A - Stones on the Table.
 *        This program calculates the minimum number of stones to remove from a row
 *        such that no two adjacent stones have the same color.
 *
 * Algorithm: Linear Scan and Greedy Removal
 *   1. Reads an integer `length`, representing the total number of stones.
 *   2. Reads the `input` string of stone colors (e.g., "RRGGBB").
 *   3. Initializes `total` to 0 (counter for stones to remove) and `state` to the
 *      color of the first stone.
 *   4. Iterates through the string from the second stone onwards:
 *      a. If the current stone's color is the same as the `state` (previous stone's color),
 *         increments `total` (this stone needs to be removed).
 *      b. If the current stone's color is different, updates `state` to the current
 *         stone's color.
 *   5. Prints the final `total`.
 *
 * Time Complexity: O(L), where L is the length of the input string `input`, due to the single linear pass.
 * Space Complexity: O(L) for storing the input string `input`. O(1) for auxiliary variables.
 */

#include <cstdio> // Required for scanf.
#include <iostream>
#include <string> // Required for std::string and getline.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Stones on the Table" problem.
   *
   * Reads the number of stones and their colors, then calculates the minimum
   * stones to remove for distinct adjacent colors.
   *
   * @return int Returns 0 upon successful execution.
   */
  int length; // Total number of stones.
  // Reads the length of the string. `scanf("%d\n", ...)` is used to consume the
  // newline character left in the buffer, which would otherwise be read by `getline`.
  scanf("%d\n", &length); 
  string input; // String representing the colors of the stones (e.g., "RRGGBB").
  getline(cin, input); // Reads the entire line for the stone colors.

  int total = 0; // Counter for the minimum number of stones to remove.
  int index = 0; // Current index in the `input` string.
  char state = input[0]; // Stores the color of the previously processed stone.

  /**
   * Block Logic: Iterates through the stones starting from the second one, comparing
   *              each stone's color with the previous one.
   * Invariant: `total` holds the count of adjacent same-colored stones found so far.
   *            `state` holds the color of the last stone that did not need to be removed.
   * Pre-condition: `input` string contains stone colors, `length` is its size.
   */
  while (++index < length) { // Loop from the second character (`index` starts at 1).
    /**
     * Conditional Logic: Checks if the current stone has the same color as the previous stone.
     */
    if (input[index] == state) {
      ++total; // If colors are the same, this stone needs to be removed.
    } else {
      state = input[index]; // If colors are different, update `state` to the current stone's color.
    }
  }
  printf("%d\n", total); // Prints the total number of stones to remove.
  return 0;
}
