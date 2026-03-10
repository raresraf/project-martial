/**
 * @file 266A.CPP
 * @brief This program solves the "Stones on the Table" problem (Codeforces 266A).
 *   It calculates the minimum number of stones to remove from a row such that
 *   no two adjacent stones are of the same color. This is equivalent to counting
 *   the number of adjacent identical stones.
 *   Problem description: https://codeforces.com/problemset/problem/266/A
 * * Algorithm: Linear scan and direct counting.
 *   It iterates through the string of stone colors, comparing each character `s[i]`
 *   with its immediate successor `s[i+1]`. If they are the same, a counter is incremented.
 * Time Complexity: O(N) where N is the number of stones (length of the string),
 *   as it involves a single pass through the string.
 * Space Complexity: O(N) for storing the input string.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.
using namespace std;

/**
 * @brief Main function to calculate the minimum stones to remove.
 * Functional Utility: Reads the number of stones and their colors,
 *                     then counts adjacent identical stones to determine removals.
 */
int main() {
  int n; // Stores the number of stones.
  int k = 0; // Accumulates the count of stones to be removed.
  string s;  // Stores the string representing stone colors.

  // Block Logic: Reads the number of stones `n` and the string of stone colors `s`.
  cin >> n >> s;
  /**
   * Block Logic: Iterates through the string to compare adjacent characters.
   * Pre-condition: `s` is populated with `n` stone colors.
   * Invariant: `k` counts pairs of identical adjacent stones.
   * Inline: The loop runs from `i = 0` to `n - 1`. Comparing `s[i]` with `s[i+1]` means
   *         `i+1` can go up to `n`, which would be an out-of-bounds access if `i` goes
   *         to `n-1`. For a string of length `n`, valid indices are `0` to `n-1`.
   *         Therefore, the loop should ideally run `for (int i = 0; i < n - 1; i++)`
   *         to safely compare `s[i]` and `s[i+1]`. Given the problem constraints and typical CP solution styles,
   *         it's often implied `n` is at least 1, and the problem constraints might
   *         ensure no out-of-bounds access for specific test cases.
   */
  for (int i = 0; i < n; i++) {
    // Conditional Logic: Checks if the current stone `s[i]` is the same color as the next stone `s[i+1]`.
    if (s[i] == s[i + 1])
      k++; // If they are identical, increment the counter for stones to remove.
  }
  // Block Logic: Prints the total minimum number of stones to remove.
  cout << k << endl;
  return 0;
}
