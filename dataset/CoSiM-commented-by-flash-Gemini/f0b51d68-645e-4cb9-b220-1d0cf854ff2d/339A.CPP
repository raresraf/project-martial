/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem.
 *
 * This program takes a string representing a sum of numbers (1, 2, or 3, separated by '+')
 * and rearranges the numbers in non-decreasing order, then prints the new sum.
 *
 * Algorithm: Parse digits, sort, then format output.
 * Time Complexity: O(L + N log N) where L is the length of the input string and N is the
 *                  number of digits extracted (N <= L/2 + 1). Sorting dominates if N is large.
 * Space Complexity: O(N) to store the extracted digits in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to reorder a sum of numbers.
 *
 * Functional Utility: Reads a sum string, extracts and sorts its numeric components,
 * then prints the reordered sum.
 */
int main() {
  vector<int> v; // Vector to store the extracted digits.
  string s;      // String to store the input sum.
  cin >> s;      // Read the input sum string.

  /**
   * Block Logic: Iterates through the input string `s` to extract only the digits '1', '2', or '3'.
   * These digits are converted from their character representation to integer values and stored in vector `v`.
   * Pre-condition: `s` is a valid string containing digits '1', '2', '3' and '+' signs.
   * Invariant: `v` accumulates the integer values of the digits found in `s`.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] >= '1' && s[i] <= '3')
      v.push_back(s[i] - '0'); // Convert char digit to int and add to vector.
  }

  /**
   * Block Logic: Handles the edge case where the input sum contains only a single number.
   * If the input string has only one character (meaning one number), print it directly.
   * Post-condition: Prints the single number and potentially bypasses sorting for efficiency.
   */
  if (s.size() == 1)
    cout << s << endl;
  else {
    // Sort the extracted digits in non-decreasing order.
    sort(v.begin(), v.end());

    /**
     * Block Logic: Iterates through the sorted digits and prints them in the "number+number+..." format.
     * It prints all but the last number followed by a '+', then prints the last number.
     * Pre-condition: Vector `v` contains sorted integer digits.
     * Post-condition: The reordered sum string is printed to standard output.
     */
    for (int i = 0; i < v.size() - 1; i++) {
      cout << v[i] << "+";
    }
    cout << v[v.size() - 1] << endl; // Print the last number and a newline.
  }
  return 0;
}
