/**
 * @file 112A.CPP
 * @brief Solution to Codeforces problem 112A - Petya and Strings.
 *        This program compares two input strings lexicographically, ignoring case differences.
 *        It outputs -1 if the first string is lexicographically smaller, 1 if it's larger,
 *        and 0 if they are equal.
 *
 * Algorithm: Lexicographical Comparison with Case Normalization
 *   1. Reads two input strings `a` and `b`.
 *   2. Converts both strings `a` and `b` to their lowercase equivalents, storing them
 *      in new strings `c` and `d` respectively. This step ensures case-insensitive comparison.
 *   3. Performs a standard lexicographical comparison between the lowercase strings `c` and `d`.
 *   4. Prints -1, 0, or 1 based on the comparison result.
 *
 * Time Complexity: O(L), where L is the length of the input strings (assuming they are of equal length).
 *                  This involves a linear pass to convert characters to lowercase and another
 *                  linear pass for the string comparison.
 * Space Complexity: O(L) for storing the two input strings (`a`, `b`) and their lowercase
 *                   versions (`c`, `d`).
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

string a, b; // Input strings.
string c, d; // Lowercase versions of input strings.

int main() {
  /**
   * @brief Main function that solves the "Petya and Strings" problem.
   *
   * Reads two strings, converts them to lowercase, and then compares them
   * lexicographically, printing the result (-1, 0, or 1).
   *
   * @return int Returns 0 upon successful execution.
   */
  cin >> a >> b; // Reads the two strings from input.
  int n = (int)(a.length()); // Gets the length of the first string (assuming both have same length).
  /**
   * Block Logic: Iterates through both input strings to convert all characters to lowercase.
   * Invariant: `c` and `d` accumulate the lowercase versions of `a` and `b` respectively.
   * Pre-condition: `a` and `b` are valid strings, `n` is their length.
   */
  for (int i = 0; i < n; i++) {
    // Conditional Logic: Converts character `a[i]` to lowercase if it's uppercase.
    if ('A' <= a[i] && a[i] <= 'Z')
      c.push_back(a[i] - 'A' + 'a'); // Converts to lowercase using ASCII arithmetic.
    else
      c.push_back(a[i]); // Appends character as is if already lowercase or not an alphabet.
    
    // Conditional Logic: Converts character `b[i]` to lowercase if it's uppercase.
    if ('A' <= b[i] && b[i] <= 'Z')
      d.push_back(b[i] - 'A' + 'a'); // Converts to lowercase using ASCII arithmetic.
    else
      d.push_back(b[i]); // Appends character as is.
  }
  
  /**
   * Conditional Logic: Performs lexicographical comparison of the lowercase strings.
   * Uses standard string comparison operators.
   */
  if (c < d)
    cout << -1; // `c` is lexicographically smaller than `d`.
  if (c == d)
    cout << 0; // `c` is lexicographically equal to `d`.
  if (c > d)
    cout << 1; // `c` is lexicographically greater than `d`.
  // Note: The problem implies that only one of these conditions will be true.
  return 0; // The problem statement does not require a newline character.
}
