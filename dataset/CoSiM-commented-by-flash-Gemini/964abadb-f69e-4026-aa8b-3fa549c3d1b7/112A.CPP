/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two input strings.
 * * Algorithm: String Transformation and Lexicographical Comparison.
 * Time Complexity: $O(L)$ where L is the length of the longer string, due to single-pass transformation and comparison.
 * Space Complexity: $O(1)$ additional space, as transformations are in-place or for storing a copy of strings.
 *
 * This program reads two strings, converts them to lowercase, and then compares them
 * lexicographically, printing 1 if the first is greater, 0 if they are equal,
 * and -1 if the first is less than the second.
 */

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming.
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and unties
   * cin from cout to optimize high-throughput I/O required for competitive programming constraints.
   */
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s, s1; // Declare two strings to store input.
  cin >> s >> s1; // Reads the two input strings.

  /**
   * Block Logic: Transforms both input strings to lowercase.
   * Pre-condition: `s` and `s1` contain input strings (potentially mixed case).
   * Invariant: After `transform`, both strings `s` and `s1` will contain only lowercase characters.
   */
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  
  /**
   * Block Logic: Performs a lexicographical comparison of the two lowercase strings.
   * Invariant: Exactly one of the three conditions (s > s1, s == s1, s < s1) will be true.
   * Post-condition: Prints 1, 0, or -1 based on the comparison result.
   */
  if (s > s1)
    cout << "1\n"; // Inline: `s` is lexicographically greater than `s1`.
  if (s == s1)
    cout << "0\n"; // Inline: `s` is lexicographically equal to `s1`.
  if (s < s1)
    cout << "-1\n"; // Inline: `s` is lexicographically less than `s1`.
  
  return 0; // Standard successful exit code for main function.
}
