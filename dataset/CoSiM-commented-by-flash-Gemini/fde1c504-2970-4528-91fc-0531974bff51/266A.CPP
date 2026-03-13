/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces (266A).
 * @details This program calculates the minimum number of stones to remove from a
 * row so that no two adjacent stones have the same color. This implementation
 * uses an efficient single-pass approach.
 *
 * @section Algorithm
 * The algorithm iterates through the sequence of stones once, keeping track of the
 * color of the last valid (non-duplicate) stone encountered.
 * 1. Read the number of stones, 'n', and the sequence of colors.
 * 2. Initialize a `temp` variable with the color of the first stone.
 * 3. Initialize a removal counter `num` to zero.
 * 4. Loop from the second stone to the end of the sequence. For each stone:
 *    a. If its color is the same as `temp`, increment the `num` counter.
 *    b. If its color is different, update `temp` to the new color.
 * 5. After the loop, `num` holds the total count of stones that must be removed.
 *
 * @section Complexity
 * - Time Complexity: O(n), where 'n' is the number of stones, as the algorithm
 *   only requires a single pass through the input.
 * - Space Complexity: O(n) to store the input string in a character array.
 */
#include <cstring>
#include <iostream>

int main(int argc, char const *argv[]) {
  /**
   * @var i Loop counter.
   * @var n The number of stones on the table.
   * @var num The total number of stones to be removed.
   * @var str A C-style string to store the sequence of stone colors.
   * @var temp A state variable to hold the color of the last non-duplicate stone.
   */
  int i, n, num = 0;
  char str[100], temp;
  std::cin >> n;
  std::cin >> str;

  // Initialize the state with the color of the first stone.
  temp = str[0];

  /**
   * @brief Block Logic: Iterates through the stones from the second position
   * to count adjacent duplicates.
   * @invariant `temp` holds the color of the last stone in the valid,
   * non-adjacent-duplicate sequence. `num` holds the count of removals so far.
   */
  for (i = 1; i < n; i++) {
    /**
     * @brief Logic: If the current stone's color matches the previous one,
     * it's a duplicate that needs to be removed. Otherwise, update the
     * `temp` color to the current stone's color for the next comparison.
     */
    if (str[i] == temp)
      num++;
    else
      temp = str[i];
  }
  
  // Output the total number of stones to remove.
  std::cout << num << '\n';
  return 0;
}