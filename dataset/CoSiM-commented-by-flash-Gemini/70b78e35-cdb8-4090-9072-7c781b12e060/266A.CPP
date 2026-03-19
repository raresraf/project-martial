/**
 * @file 266A.CPP
 * @brief Solution to Codeforces Problem 266A: "Stones on the Table".
 * Counts the minimum number of stones to remove to ensure no two adjacent stones
 * are the same color. This is equivalent to counting adjacent pairs of same-colored stones.
 * Algorithm: Iterates through the sequence of stones and increments a counter
 * for every adjacent pair of identical colors. This is an efficient single-pass approach.
 * Time Complexity: O(N), where N is the number of stones.
 * Space Complexity: O(N) to store the string of stone colors.
 */
#include <algorithm> // Included but not explicitly used in this solution.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <string>    // Required for string manipulation.
using namespace std;

/**
 * @brief Main function to solve the "Stones on the Table" problem.
 * Reads the number of stones and their colors, then calculates and prints
 * the minimum number of stones to remove.
 */
int main() {
  int n;      // Stores the total number of stones.
  int count = 0; // Counter for the number of stones that need to be removed.
  cin >> n;   // Reads the total number of stones.
  string str; // Stores the sequence of stone colors as a string.
  cin >> str; // Reads the string of stone colors.

  /**
   * Block Logic: Iterates through the string of stone colors to find adjacent duplicates.
   * Pre-condition: 'str' contains 'n' characters representing stone colors. 'count' is 0.
   * Invariant: After each iteration, 'count' holds the number of adjacent duplicates
   *            found up to the current position 'i'.
   */
  for (int i = 1; i < n; i++) {
    /**
     * Block Logic: Compares the current stone's color with the previous stone's color.
     * If they are the same, it means one of them needs to be removed.
     */
    if (str[i] == str[i - 1]) {
      count++; // Increment the counter for removable stones.
    }
  }
  std::cout << count; // Prints the total number of stones to be removed.
  return 0;
}