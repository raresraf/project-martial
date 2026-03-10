/**
 * @file 266A.CPP
 * @brief This program solves the "Stones on the Table" problem (Codeforces 266A).
 *   It calculates the minimum number of stones to remove from a row such that
 *   no two adjacent stones are of the same color. This is equivalent to counting
 *   the number of adjacent identical stones.
 *   Problem description: https://codeforces.com/problemset/problem/266/A
 * * Algorithm: Linear scan and direct counting.
 *   It iterates through the string of stone colors, comparing each character `col[i]`
 *   with its immediate successor `col[i+1]`. If they are the same, a counter is incremented.
 * Time Complexity: O(N) where N is the number of stones (length of the string `col`),
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
  int x; // Stores the number of stones (length of the string).
  // Block Logic: Reads the total number of stones `x`.
  cin >> x;
  string col; // Stores the string representing stone colors.
  // Block Logic: Reads the string of stone colors `col`.
  cin >> col;
  int coun = 0; // Accumulates the count of stones to be removed.
  /**
   * Block Logic: Iterates through the string to compare adjacent characters.
   * Pre-condition: `col` is populated with `x` stone colors.
   * Invariant: `coun` counts pairs of identical adjacent stones.
   * The loop runs from `i = 0` to `x - 2`, ensuring `col[i+1]` is always within bounds.
   */
  for (int i = 0; i + 1 < x; i++) {
    // Conditional Logic: Checks if the current stone `col[i]` is the same color as the next stone `col[i+1]`.
    if (col[i] == col[i + 1]) {
      coun++; // If they are identical, increment the counter for stones to remove.
    }
  }
  // Block Logic: Prints the total minimum number of stones to remove.
  cout << coun;
  return 0;
}
