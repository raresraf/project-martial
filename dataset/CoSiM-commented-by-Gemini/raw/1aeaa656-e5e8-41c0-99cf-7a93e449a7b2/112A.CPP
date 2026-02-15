/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings and compares them character by character,
 * ignoring case. It outputs -1 if the first string is lexicographically smaller,
 * 1 if it is lexicographically larger, and 0 if they are equal.
 *
 * @algorithm Case-insensitive lexicographical comparison.
 * 1. Read two strings.
 * 2. Iterate through both strings simultaneously.
 * 3. For each pair of characters, convert them to lowercase.
 * 4. Compare the lowercase characters.
 * 5. If a difference is found, determine the result (-1 or 1) and stop.
 * 6. If the end of the strings is reached without any difference, they are equal (0).
 *
 * @complexity
 * Time: O(N), where N is the length of the strings. The loop runs at most N times.
 * Space: O(N) to store the two input strings.
 */
#include <iostream>
// Includes <string> and <cctype> for std::string, tolower
#include <string>
#include <cctype>

using namespace std;

int main() {
  // @var s, s1 The two strings to be compared.
  string s, s1;
  cin >> s >> s1;
  // @var val This variable is incremented when characters are equal but its value is never used. It is redundant.
  int val = 0;
  /**
   * @var res Stores the result of the comparison.
   * - 0: Strings are equal so far.
   * - 1: First string is lexicographically greater.
   * - -1: First string is lexicographically smaller.
   */
  int res = 0;

  /**
   * @brief Iterates through the strings, comparing characters case-insensitively.
   * The loop breaks as soon as the first lexicographical difference is found.
   */
  for (int i = 0; i < s.size(); i++) {
    // Convert characters to lowercase for case-insensitive comparison.
    char ch = tolower(s[i]);
    char ch1 = tolower(s1[i]);

    if (ch == ch1) {
      // Characters are the same, continue checking.
      val++;
    } else if (ch > ch1) {
      // First string is lexicographically greater.
      res = 1;
      break;
    } else {
      // First string is lexicographically smaller.
      res = -1;
      break;
    }
  }

  /**
   * @brief Prints the final comparison result.
   * If `res` is non-zero, it means a difference was found and stored in `res`.
   * If the loop finished without finding any difference, `res` remains 0,
   * indicating the strings are equal.
   */
  if (res) {
    cout << res;
  } else {
    cout << "0";
  }
  return 0;
}