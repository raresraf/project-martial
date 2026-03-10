/**
 * @file 112A.CPP
 * @brief This program solves the "Petya and Strings" problem (Codeforces 112A).
 *   It compares two input strings lexicographically, ignoring case.
 *   Problem description: https://codeforces.com/problemset/problem/112/A
 * * Algorithm: Case-insensitive lexicographical comparison.
 *   1. Reads two input strings.
 *   2. Converts all characters in both strings to lowercase.
 *   3. Compares the resulting lowercase strings using standard lexicographical comparison.
 *   4. Prints -1 if the first string is smaller, 1 if the second is smaller, and 0 if they are equal.
 * Time Complexity: O(L) where L is the length of the strings (assuming they are of similar length).
 *   This is due to iterating through the strings once for lowercase conversion and once for comparison.
 * Space Complexity: O(L) to store the input strings.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.
#include <algorithm> // Required for std::tolower (though cctype is usually for single chars).
#include <cctype>   // Required for std::tolower.

using namespace std;

/**
 * @brief Main function to perform case-insensitive string comparison.
 * Functional Utility: Reads two strings, converts them to lowercase,
 *                     compares them lexicographically, and prints the comparison result (-1, 0, or 1).
 */
int main() {
  string s1, s2; // `s1` and `s2` store the two input strings.

  // Block Logic: Reads the two input strings.
  cin >> s1 >> s2;

  // Assumes both strings have the same length as per typical problem constraints,
  // or takes the length of the first string for iteration.
  int length = s1.length();

  /**
   * Block Logic: Converts all characters in both `s1` and `s2` to lowercase.
   * Pre-condition: `s1` and `s2` contain characters that can be converted to lowercase.
   * Invariant: After this loop, both strings `s1` and `s2` will contain only lowercase characters.
   * Inline: `static_cast<char>(tolower(s1[i]))` converts the character to lowercase
   *         and casts it back to `char`, as `tolower` returns an `int`.
   */
  for (int i = 0; i < length; i++) {
    s1[i] = static_cast<char>(tolower(s1[i]));
    s2[i] = static_cast<char>(tolower(s2[i]));
  }

  /**
   * Conditional Logic: Performs lexicographical comparison of the lowercase strings.
   * String comparison operators (`<`, `>`, `==`) in C++ perform lexicographical comparison.
   * - If `s1` is lexicographically smaller than `s2`, prints -1.
   * - If `s1` is lexicographically larger than `s2`, prints 1.
   * - If `s1` and `s2` are equal, prints 0.
   */
  if (s1 < s2) {
    cout << "-1" << endl;
  } else if (s1 > s2) {
    cout << "1" << endl;
  } else {
    cout << "0" << endl;
  }

  return 0;
}
