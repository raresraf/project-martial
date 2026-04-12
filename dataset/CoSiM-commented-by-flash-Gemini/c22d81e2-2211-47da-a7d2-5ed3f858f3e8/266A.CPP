/**
 * @file 266A.CPP
 * @brief Calculates the minimum number of stones to remove to ensure no adjacent stones have the same color.
 *   This program processes a sequence of colored stones (represented as characters in a string)
 *   and determines how many stones need to be removed to make all adjacent stones different.
 * * Algorithm: Single-Pass Scan and Count.
 *   The algorithm iterates through the sequence of stones, comparing each stone with its
 *   predecessor. If two adjacent stones are of the same color, one of them must be removed.
 *   The count of such removals is accumulated.
 * Time Complexity: O(N) where N is the number of stones (length of the string).
 * Space Complexity: O(N) for storing the input string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to determine the minimum stone removals.
 * Functional Utility: Handles input/output, reads the stone sequence, and calculates the removals.
 */
int main(void) {
  // Functional Utility: Decouples C++ streams from standard C streams and flushes
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  int n; // Number of stones.
  string s; // String representing the colors of the stones.
  cin >> n >> s;

  int ans = 0; // Counter for the number of stones to remove.

  int i = 1; // Index to iterate through the string, starting from the second stone.

  /**
   * Block Logic: Iterates through the string to find adjacent stones of the same color.
   * Invariant: 'i' points to the current stone being examined.
   */
  while (i < n) {
    /**
     * Block Logic: Counts consecutive stones of the same color.
     * Pre-condition: s[i-1] and s[i] are potentially the same color.
     * Invariant: 'ans' is incremented for each stone that needs to be removed from a consecutive block.
     */
    while (s[i - 1] == s[i]) {
      ++i;   // Move to the next stone.
      ++ans; // Increment the count of stones to remove.
    }
    ++i; // Move to the next stone after a block of same-colored stones has been processed.
  }

  // Outputs the total minimum number of stones to remove.
  cout << ans << "\n";

  return 0;
}