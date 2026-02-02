/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A - Petya and Strings.
 *
 * This problem involves comparing two strings lexicographically, but in a case-insensitive manner.
 * The task is to determine if the first string is lexicographically less than, greater than, or equal to the second string,
 * ignoring case.
 *
 * Algorithm: Case-insensitive lexicographical comparison.
 * The code reads two input strings, converts both to uppercase (or lowercase, the choice is arbitrary as long as consistent)
 * to ensure case-insensitivity, and then performs a standard lexicographical comparison.
 *
 * Time Complexity: O(L), where L is the length of the shorter string (due to `transform` and string comparison).
 * Space Complexity: O(L) for storing the strings.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope, simplifying code by avoiding explicit std:: prefixes.

/**
 * @brief Main function: Solves the Petya and Strings problem.
 * Functional Utility: Reads two strings, converts them to uppercase for a case-insensitive
 * comparison, and then prints -1, 1, or 0 based on their lexicographical order.
 * Algorithm: Case-insensitive string comparison.
 */
int main() {
  int m = 0, n = 0; // Declares integers `m` and `n`, unused in this specific solution.
  string a, b; // Declares two strings `a` and `b` for input.
  cin >> a >> b; // Reads the two input strings.

  // Functional Utility: Converts all characters in string `a` to uppercase for case-insensitive comparison.
  // Precondition: `a` is a valid string.
  // Postcondition: All alphabetic characters in `a` are converted to their uppercase equivalents.
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  // Functional Utility: Converts all characters in string `b` to uppercase for case-insensitive comparison.
  // Precondition: `b` is a valid string.
  // Postcondition: All alphabetic characters in `b` are converted to their uppercase equivalents.
  transform(b.begin(), b.end(), b.begin(), ::toupper);

  // Block Logic: Perform lexicographical comparison on the uppercase strings and print results.
  // The standard string comparison operator (`<`, `>`) performs lexicographical comparison.
  if (a < b)
    cout << -1 << endl; // If string `a` is lexicographically less than `b`.
  else if (a > b)
    cout << 1 << endl; // If string `a` is lexicographically greater than `b`.
  else
    cout << 0 << endl; // If strings `a` and `b` are lexicographically equal.
  return 0; // Indicate successful program execution.
}
