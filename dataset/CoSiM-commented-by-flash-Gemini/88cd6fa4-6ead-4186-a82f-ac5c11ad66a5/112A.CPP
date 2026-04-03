/**
 * @file 112A.CPP
 * @brief Performs case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to lowercase, and then determines
 * their lexicographical order. It outputs -1 if the first string is smaller,
 * 1 if the second string is smaller, and 0 if they are equal. This is functionally
 * equivalent to the POSIX `strcasecmp` function.
 *
 * @algorithm Single-pass character-by-character comparison.
 * @time_complexity O(L), where L is the length of the input strings, as it stops at the first differing character.
 * @space_complexity O(L1 + L2) to store the two input strings.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Converts an uppercase ASCII character to lowercase.
 * @param ch The character to convert.
 * @return The lowercase version of the character if it was uppercase, otherwise the character itself.
 * @note This is a custom implementation; the standard library provides `std::tolower`.
 */
char tolower(char ch) {
  if (ch >= 'A' && ch <= 'Z')

  {
    // Inline: Converts uppercase to lowercase by adding the ASCII difference.
    return ch - ('Z' - 'z');
  }
  return ch;
}

int main() {
  string s1, s2;
  // @brief Flag to indicate the comparison result. 0 for equal, 1 for different.
  int flag = 0, i;

  // Functional Utility: Reads two strings from standard input.
  cin >> s1;
  cin >> s2;

  /**
   * @brief Block Logic: Iterates through the strings to find the first lexicographical difference.
   *
   * This loop compares characters at each position after converting them to lowercase.
   * It terminates at the first mismatch or at the end of the string.
   * @invariant `i` is the current index being compared.
   */
  for (i = 0; i < s1.length(); i++) {
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);

    // Block Logic: Checks for the first character mismatch.
    // If a difference is found, the flag is set and the loop is exited.
    if (s1[i] != s2[i]) {
      flag = 1;
      break;
    }
  }

  /**
   * @brief Block Logic: Determines and prints the final comparison result.
   *
   * Based on the loop's result, this block prints the lexicographical relationship
   * between the two strings.
   * @pre-condition The loop has completed, and `i` holds the index of the first difference, if any.
   */
  if (flag == 0)
    // The strings were identical.
    cout << "0";
  else {
    // The strings differed at index `i`.
    if (s1[i] < s2[i])
      cout << "-1";
    else
      cout << "1";
  }
  return 0;
}