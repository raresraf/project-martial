/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A - Stones on the Table.
 *
 * This problem involves removing stones from a row to ensure no two adjacent stones are the same color.
 * The goal is to find the minimum number of stones to remove. The provided solution aims to achieve this
 * by iterating through the string and removing a stone if it's the same color as an adjacent stone.
 *
 * Algorithm: Greedy iterative removal.
 * The code iterates through the string, and if a stone `s[i]` has the same color as `s[i-1]`,
 * or `s[i]` has the same color as `s[i+1]` (which is effectively removing `s[i]` to break the adjacency),
 * it increments a counter and removes `s[i]`. The loop index `i` is decremented to re-check the new `s[i]`
 * after removal, as the adjacent characters might have changed.
 *
 * Time Complexity: O(N * L) in the worst case, where N is the original string length and L is the average string length after removals.
 * This is because `s.erase()` operation can take O(L) time. A more optimized approach (single pass without erase) would be O(N).
 * Space Complexity: O(N) for storing the string.
 */

#include <bits/stdc++.h>
#define endl "\n" 
#define lli long long int 
#define ulli unsigned long long int 

using namespace std;

/**
 * @brief Main function: Solves the Stones on the Table problem.
 * Functional Utility: Reads a string representing a row of stones and calculates
 * the minimum number of stones to remove to satisfy the condition that no two
 * adjacent stones are the same color.
 * Algorithm: Iterative removal of problematic stones.
 */
int main() {
  /**
   * Functional Utility: Optimizes C++ standard streams for faster input/output operations.
   * `ios_base::sync_with_stdio(false)` unties C++ streams from C standard streams.
   * `cin.tie(0)` unties `cin` from `cout`, preventing flushes before input operations.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  int n; 
  cin >> n;

  string s;
  cin >> s;

  int count = 0;
  /**
   * Block Logic: Iterate through the string to find and remove adjacent stones of the same color.
   * Precondition: `s` is a valid string of characters representing stone colors.
   * Invariant: `count` stores the number of stones removed so far. `s` is modified in place.
   *            The loop index `i` adjusts for removals to re-evaluate adjacent characters.
   * Logic: The loop starts from the second character (`i=1`) and goes up to the second-to-last character (`s.size()-1`)
   *        to safely check `s[i-1]` and `s[i+1]`.
   */
  for (int i = 1; i < s.size() - 1; i++) {
    /**
     * Block Logic: Check if the current stone `s[i]` has the same color as an adjacent stone.
     * Logic: The condition `!(s[i] != s[i-1] && s[i] != s[i+1])` is true if `s[i]` is equal to `s[i-1]` OR `s[i]` is equal to `s[i+1]`.
     *        This implies `s[i]` is a stone that creates an adjacency violation.
     */
    if (!(s[i] != s[i - 1] && s[i] != s[i + 1])) {
      count++; 
      s.erase(s.begin() + i); 
      i--; 
    }
  }

  /**
   * Block Logic: Handle a specific edge case for a string of size 2 after the main loop.
   * Precondition: The main loop has completed.
   * Logic: If the remaining string has exactly two characters and they are of the same color,
   *        one more removal is needed.
   */
  if (s[0] == s[1] && s.size() == 2)
    count++; 

  cout << count << endl;
  return 0;
}
