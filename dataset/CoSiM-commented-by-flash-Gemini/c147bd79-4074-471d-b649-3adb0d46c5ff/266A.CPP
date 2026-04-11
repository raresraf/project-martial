/**
 * @file 266A.CPP
 * @brief This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color.
 *
 * Algorithm: Greedy approach. Iterate through the stones and count adjacent identical colors.
 * Time Complexity: O(L) where L is the length of the string of stones.
 * Space Complexity: O(L) to store the string of stones.
 */

#include <iostream>
#include <string> // Required for std::string

using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of stones in the row (though not directly used after reading).
  int n = 0;
  // Functional Utility: 'stones' counts the number of adjacent identical stones that need to be removed.
  int stones = 0;
  // Functional Utility: 'color' stores the string representing the colors of the stones.
  string color = "";

  // Block Logic: Reads the number of stones 'n' (integer) from standard input.
  cin >> n;
  // Block Logic: Reads the string of stone colors from standard input.
  cin >> color;

  /**
   * Block Logic: Iterates through the string of stone colors to find adjacent identical colors.
   * Invariant: 'i' represents the current index of the stone being examined. The loop continues
   *            until the second to last stone, as it checks 'color[i]' and 'color[i+1]'.
   */
  for (int i = 0; i < color.length(); i++) {
    /**
     * Block Logic: Checks if the current stone and the next stone have the same color.
     * Pre-condition: 'i+1' is a valid index within the 'color' string.
     * Invariant: If adjacent stones are identical, 'stones' counter is incremented.
     */
    if (color[i] == color[i + 1]) {
      stones++;
    }
  }

  // Output the total number of stones that need to be removed.
  cout << stones;
  return 0;
}
