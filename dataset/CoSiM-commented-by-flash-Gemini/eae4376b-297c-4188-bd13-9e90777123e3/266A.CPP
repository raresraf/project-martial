/**
 * @file 266A.CPP
 * @brief This program solves the "Stones on the Table" problem (Codeforces 266A).
 *   It calculates the minimum number of stones to remove from a row such that
 *   no two adjacent stones are of the same color.
 *   Problem description: https://codeforces.com/problemset/problem/266/A
 * * Algorithm: Linear scan with previous character tracking.
 *   It iterates through the string of stone colors, comparing each stone to the one
 *   immediately preceding it. If they are the same color, a stone needs to be removed,
 *   and a counter is incremented.
 * Time Complexity: O(N) where N is the number of stones (length of the string),
 *   as it involves a single pass through the string.
 * Space Complexity: O(N) for storing the input string.
 */
// 266A.cpp
// Stones on the Table
// Author: Ikki
// May 31, 2015

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
  // Block Logic: Reads the total number of stones.
  cin >> n;
  string str; // Stores the string representing stone colors.
  // Block Logic: Reads the string of stone colors.
  cin >> str;
  int ans = 0; // Accumulates the count of stones to be removed.
  // `prev` stores the color of the previous stone, initialized with the first stone's color.
  char prev = str[0];
  /**
   * Block Logic: Iterates through the string of stone colors starting from the second stone.
   * Pre-condition: `str` is populated with stone colors, `prev` holds the first stone's color.
   * Invariant: `ans` counts adjacent identical stones, and `prev` always holds the color
   *            of the last unique stone encountered.
   */
  for (int i = 1; i < str.size(); i++) {
    // Conditional Logic: If the current stone has the same color as the previous one,
    // it needs to be removed, so increment the counter.
    if (str[i] == prev)
      ans++;
    // Conditional Logic: If the current stone has a different color, it becomes the new
    // reference for comparison with the next stone.
    else
      prev = str[i];
  }
  // Block Logic: Prints the total minimum number of stones to remove.
  cout << ans << endl;
  return 0;
}