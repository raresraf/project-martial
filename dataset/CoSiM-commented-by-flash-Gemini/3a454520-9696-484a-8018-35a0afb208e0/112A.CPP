/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A - Petya and Strings.
 *
 * This program performs a case-insensitive lexicographical comparison of two input strings.
 * It determines which string comes first alphabetically or if they are identical,
 * outputting 1, -1, or 0 respectively (where 1 means the first string is greater,
 * -1 means it's smaller, and 0 means they are equal).
 *
 * Algorithm: Full lowercase conversion then direct string comparison. Both strings
 * are entirely converted to lowercase first. Then, C++'s built-in string comparison
 * operators (>, <, ==) are used to perform the lexicographical comparison.
 * Time Complexity: O(L), where L is the length of the strings, as each character
 *                  is processed once for conversion and then implicitly for comparison.
 * Space Complexity: O(L) for storing the input strings.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Required for std::string and its methods.
#include <algorithm> // Required for std::tolower (though it's often included via <bits/stdc++.h> or can use cctype).

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Petya and Strings problem logic.
   *
   * Reads two strings from standard input. It converts both strings entirely
   * to lowercase to ensure case-insensitive comparison. Then, it uses direct
   * string comparison operators to determine their lexicographical order and
   * prints 1, -1, or 0 based on the result.
   */
  // Declare two string variables to hold the input lines, initialized as empty.
  string s1 = "";
  string s2 = "";
  // Declare 'result' to store the comparison outcome (0, 1, or -1).
  int result = 0;
  // Read the two input strings.
  cin >> s1 >> s2;

  // Block Logic: Convert both strings to lowercase for case-insensitive comparison.
  // Invariant: 'i' iterates through the strings.
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = tolower(s1[i]); // Convert character in s1 to lowercase.
    s2[i] = tolower(s2[i]); // Convert character in s2 to lowercase.
  }

  // Block Logic: Perform lexicographical comparison using string operators.
  // Pre-condition: Both strings are now in lowercase.
  // If strings are equal.
  if (s1 == s2) {
    result = 0; // Set result to 0.
  }
  // If string s1 is lexicographically greater than s2.
  else if (s1 > s2) {
    result = 1; // Set result to 1.
  }
  // If string s1 is lexicographically less than s2.
  else if (s1 < s2) {
    result = -1; // Set result to -1.
  }

  // Output the comparison result, followed by a newline.
  cout << result << endl;

  return 0; // Indicate successful program execution.
}
