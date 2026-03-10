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
#include <algorithm> // Included but not directly used in the core logic.
#include <cmath>     // Included but not directly used.
#include <iostream>  // Required for std::cin, std::cout.
#include <string>    // Required for std::string.
using namespace std;

/**
 * @brief Main function to calculate the minimum stones to remove.
 * Functional Utility: Reads the number of stones and their colors,
 *                     then counts adjacent identical stones to determine removals.
 */
int main() {
  int n; // Stores the number of stones.
  int cnt = 0; // Accumulates the count of stones to be removed.
  string s;    // Stores the string representing stone colors.

  // Block Logic: Reads the number of stones `n`.
  cin >> n;

  // Block Logic: Reads the string of stone colors `s`.
  cin >> s;

  /**
   * Block Logic: Iterates through the string to compare adjacent characters.
   * Pre-condition: `s` is populated with `n` stone colors.
   * Invariant: `cnt` counts pairs of identical adjacent stones.
   * The loop runs from `i = 0` to `n - 2`, ensuring `s[i+1]` is always within bounds.
   */
  for (int i = 0; i < n - 1; i++) {
    // Conditional Logic: Checks if the current stone `s[i]` is the same color as the next stone `s[i+1]`.
    if (s[i] == s[i + 1])
      cnt++; // If they are identical, increment the counter for stones to remove.
  }
  // Block Logic: Prints the total minimum number of stones to remove.
  cout << cnt;
  return 0;
}