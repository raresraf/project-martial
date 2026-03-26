/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical string comparison.
 * @details This program reads two strings, converts them to lowercase, and then
 * compares them lexicographically. It outputs -1 if the first string is less than
 * the second, 1 if the first string is greater than the second, and 0 if they are equal.
 *
 * @algorithm Lexicographical Comparison
 * @complexity
 * - Time: O(L), where L is the length of the longest string, due to string conversion and comparison.
 * - Space: O(L), as new lowercase strings are created for comparison.
 */

#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Converts a string to its lowercase equivalent.
 * @param s The input string.
 * @return The lowercase version of the string.
 *
 * @details This function iterates through each character of the string.
 * It assumes an ASCII-like character set where uppercase letters have smaller
 * character codes than their lowercase counterparts.
 */
string toLowerCase(string s) {
  /**
   * @block Pre-condition: 's' is a string.
   * This loop iterates through each character of the string to convert it to lowercase if it's uppercase.
   * @invariant All characters up to index i-1 have been converted to lowercase where applicable.
   */
  for (size_t i = 0; i < s.size(); ++i) {
    /**
     * @block Logic: This condition checks if a character is uppercase by comparing its ASCII value.
     * `s[i] - 'a' < 0` is a non-standard way to check for uppercase, assuming that all uppercase
     * letters have ASCII values smaller than 'a'.
     */
    if (s[i] - 'a' < 0)
      s[i] += 32; // Adds 32 to convert from ASCII uppercase to lowercase.
  }
  return s;
}

/**
 * @brief Compares two strings lexicographically, ignoring case.
 * @param first The first string for comparison.
 * @param second The second string for comparison.
 * @return -1 if the first string is smaller, 1 if it's larger, and 0 if they are equal.
 */
int compare(string first, string second) {
  // Convert both strings to lowercase for a case-insensitive comparison.
  first = toLowerCase(first);
  second = toLowerCase(second);
  
  /**
   * @block Logic: Performs the lexicographical comparison on the normalized strings.
   * C++ string relational operators perform this comparison character by character.
   */
  if (first < second)
    return -1;
  return first == second ? 0 : 1;
}

int main() {
  string first, second;
  // Read the two strings from standard input.
  cin >> first >> second;
  // Output the result of the case-insensitive comparison.
  cout << compare(first, second) << endl;
  return 0;
}
