/**
 * @file 266A.CPP
 * @brief Solution for Codeforces 266A - Stones on the Table.
 * @see https://codeforces.com/problemset/problem/266/A
 *
 * @brief This program calculates the minimum number of stones to remove from a row
 * on a table such that no two adjacent stones have the same color. The colors
 * are represented by characters in a string.
 *
 * Algorithm:
 * - Read the number of stones `n` and the string `s` representing the colors of the stones.
 * - Initialize a counter `count` to 0.
 * - Iterate through the string from the second stone (index 1) to the end.
 * - In each iteration, compare the color of the current stone `s[i]` with the color
 *   of the previous stone `s[i-1]`.
 * - If they are the same, it means one of them needs to be removed to satisfy the
 *   condition. Increment the `count`.
 * - After the loop, `count` will hold the total number of stones that need to be
 *   removed.
 * - Print the final count.
 *
 * Time Complexity: O(N), where N is the number of stones. The program iterates
 * through the string once.
 * Space Complexity: O(N) to store the input string representing the stones.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  string s;
  int count = 0;

  // Read the number of stones and their colors.
  cin >> n >> s;

  /**
   * @brief Loop through the stones to find adjacent stones of the same color.
   * Invariant: `count` holds the number of stones to be removed from the
   * first `i` stones.
   */
  for (int i = 1; i < n; i++) {
    /**
     * @brief Check if the current stone has the same color as the previous one.
     * Pre-condition: `i` is always greater than 0, so `s[i-1]` is a valid access.
     * Logic: If two adjacent stones have the same color, one must be removed.
     * We increment the counter for each such pair found.
     */
    if (s[i] == s[i - 1]) {
      count++;
    }
  }

  // Output the total number of stones to remove.
  cout << count;

  return 0;
}