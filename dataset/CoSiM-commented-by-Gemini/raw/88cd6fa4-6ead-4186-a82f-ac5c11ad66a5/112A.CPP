/**
 * @file 112A.CPP
 * @brief Lexicographical comparison of two strings, ignoring case.
 *
 * @details
 * This program reads two strings and compares them lexicographically, treating
 * uppercase and lowercase letters as equivalent. It outputs -1 if the first
 * string is less than the second, 1 if the first string is greater than the
 * second, and 0 if they are equal. This is functionally equivalent to a
 * case-insensitive `strcmp`.
 *
 * @section
 * Algorithm: Single-pass character-by-character comparison after case normalization.
 * Time Complexity: O(L), where L is the length of the shorter string, as the comparison stops at the first differing character.
 * Space Complexity: O(L1 + L2), for storing the two input strings.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Converts an uppercase ASCII character to lowercase.
 * @param ch The character to convert.
 * @return The lowercase equivalent of the character if it is an uppercase letter; otherwise, the character itself.
 *
 * @note This implementation relies on the contiguous nature of ASCII character codes.
 * The expression `ch - ('Z' - 'z')` is equivalent to `ch + 32` for ASCII,
 * which correctly maps an uppercase letter to its lowercase counterpart.
 */
char tolower(char ch) {
  // Pre-condition: Check if 'ch' is within the uppercase alphabet range.
  if (ch >= 'A' && ch <= 'Z')

  {
    // Invariant: 'Z' - 'z' is a constant difference between uppercase and lowercase ASCII values.
    return ch - ('Z' - 'z');
  }
  return ch;
}

int main() {
  string s1, s2;
  int flag = 0, i; // flag is not strictly necessary but used to track equality post-loop.
  cin >> s1;
  cin >> s2;

  /**
   * @brief Iterates through both strings to find the first point of difference, ignoring case.
   *
   * Logic: The loop proceeds as long as characters are identical after being
   * converted to lowercase. It terminates at the end of the string or at the
   * first character that differs.
   */
  for (i = 0; i < s1.length(); i++) {
    // Normalize characters to lowercase for comparison.
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);

    // Pre-condition: If characters are not equal, they determine the lexicographical order.
    if (s1[i] != s2[i]) {
      flag = 1; // Mark that a difference has been found.
      break;    // Exit the loop as the result is now determined.
    }
  }

  /**
   * @brief Determines and prints the comparison result based on the loop's outcome.
   */
  if (flag == 0) // Invariant: If flag is 0, no differences were found, so strings are equal.
    cout << "0";
  else {
    // The loop was broken at index 'i', where the characters differ.
    if (s1[i] < s2[i])
      cout << "-1";
    else
      cout << "1";
  }
  return 0;
}
