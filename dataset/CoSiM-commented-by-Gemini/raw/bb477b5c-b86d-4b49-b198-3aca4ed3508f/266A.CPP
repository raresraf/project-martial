/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem.
 *
 * @details This program calculates the minimum number of stones to take from a row
 * on a table so that no two adjacent stones have the same color.
 *
 * @section
 * Algorithm:
 * The program iterates through the sequence of stones from the first to the second-to-last stone.
 * In each step, it compares the color of the current stone with the color of the next one.
 * If they are the same, a counter is incremented. This counter represents the total number
 * of stones that need to be removed.
 *
 * Time Complexity: O(N), where N is the number of stones, as it involves a single pass.
 * Space Complexity: O(N), to store the string of stone colors.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, cnt = 0;
  string s;
  cin >> n;

  cin >> s;

  /**
   * @brief This loop iterates through the stones to find adjacent pairs of the same color.
   * Invariant: 'cnt' holds the number of removals required for the stones processed so far.
   * The loop runs up to n-1 to prevent out-of-bounds access when checking s[i+1].
   */
  for (int i = 0; i < n - 1; i++) {
    /**
     * @brief Checks if a stone and its immediate neighbor have the same color.
     * Logic: If two adjacent stones are identical, one must be removed. The counter 'cnt'
     * is incremented to track the minimum number of removals.
     */
    if (s[i] == s[i + 1])
      cnt++;
  }
  cout << cnt;
  return 0;
}