/**
 * @file 112A.CPP
 * @brief This competitive programming solution performs a case-insensitive
 *        lexicographical comparison between two input strings.
 *        It outputs 1 if the first string is lexicographically greater,
 *        -1 if the second string is greater, and 0 if they are equal.
 * Algorithm: Explicit Character-by-Character Case Normalization and Comparison.
 *            1. Read two input strings.
 *            2. Iterate through each string, converting every character to uppercase
 *               (or lowercase) to ensure case-insensitive comparison.
 *            3. Iterate character by character through the normalized strings,
 *               comparing them at each position until a difference is found or
 *               the end of the shorter string is reached.
 * Time Complexity: O(L), where L is the maximum length of the two input strings.
 *                  Each string is traversed twice (once for case conversion, once for comparison).
 * Space Complexity: O(L) for storing the two input strings. The case conversion
 *                   is performed in-place.
 */
// By archit-dev, contest: Codeforces Beta Round #85 (Div. 2 Only), problem: (A)
// Petya and Strings, Accepted, #, Copy
#include <algorithm> // Not explicitly used for transform here, but generally useful.
#include <bits/stdc++.h> // Includes various standard library headers, including iostream, string, cctype.
#include <string>    // Explicitly included for std::string.
#include <cctype>    // For toupper() function.
using namespace std;

/**
 * @brief Main function of the program. Reads two strings, normalizes their case, compares them, and prints the result.
 * Functional Utility: Serves as the entry point for solving the case-insensitive string comparison problem.
 */
int main() {
  int flag = 0;   // Flag to store the comparison result: 0 for equal, 1 for s1 > s2, -1 for s1 < s2.
  string s1, s2;  // Strings to store the two input words.
  cin >> s1 >> s2; // Reads the two strings from standard input.

  // Block Logic: Converts each character of string s1 to its uppercase equivalent.
  // Invariant: After this loop, string s1 contains only uppercase characters.
  for (int i = 0; i < s1.size(); i++) {
    s1[i] = toupper(s1[i]); // Converts character at index i to uppercase.
  }
  // Block Logic: Converts each character of string s2 to its uppercase equivalent.
  // Invariant: After this loop, string s2 contains only uppercase characters.
  for (int i = 0; i < s2.size(); i++) {
    s2[i] = toupper(s2[i]); // Converts character at index i to uppercase.
  }

  // Block Logic: Compares the two uppercase strings character by character.
  // Invariant: If a difference is found, `flag` is set and the loop breaks.
  for (int i = 0; i < s1.size(); i++) {
    // If character in s1 is greater than character in s2, s1 is lexicographically greater.
    if (s1[i] > s2[i]) {
      flag = 1;  // Set flag to 1.
      break;     // Exit loop as result is determined.
    }
    // Else if character in s1 is less than character in s2, s2 is lexicographically greater.
    else if (s1[i] < s2[i]) {
      flag = -1; // Set flag to -1.
      break;     // Exit loop as result is determined.
    }
    // Else (characters are equal), continue to the next character.
    else {
      continue;
    }
  }
  cout << flag << "
"; // Prints the final comparison result followed by a newline.
  return 0;             // Indicates successful program termination.
}
