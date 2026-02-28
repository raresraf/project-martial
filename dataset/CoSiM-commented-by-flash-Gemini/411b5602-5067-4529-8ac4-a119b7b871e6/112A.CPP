/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two input strings.
 *
 * This program reads two strings from standard input. It then compares them
 * character by character, treating uppercase and lowercase letters as equivalent.
 * The comparison results in:
 * - -1 if the first string is lexicographically smaller than the second.
 * - 1 if the first string is lexicographically larger than the second.
 * - 0 if the strings are lexicographically equal.
 *
 * Algorithm: Character-by-character comparison with explicit case conversion.
 * Time Complexity: O(L) where L is the length of the shorter of the two input strings.
 *                  In the worst case (strings are equal or differ at the last character),
 *                  it iterates through the length of the strings once.
 * Space Complexity: O(1) beyond the storage required for the input strings themselves,
 *                   as only a few scalar variables are used for character storage and flags.
 */

#include <iostream>
#include <math.h> // Not strictly necessary for this program, but included in original.
using namespace std;

int main() {
  /**
   * @brief Main function to orchestrate case-insensitive string comparison.
   *
   * Initializes comparison flag, reads two strings, iterates through them
   * to perform character-by-character comparison (after converting to uppercase),
   * and then prints the comparison result.
   */

  // @param ch1 Stores the current character from the first string, converted to uppercase.
  // @param ch2 Stores the current character from the second string, converted to uppercase.
  char ch1, ch2;
  // @param s1 Stores the first input string.
  // @param s2 Stores the second input string.
  string s1, s2;
  // @param f Flag to store the comparison result: -1 for s1 < s2, 1 for s1 > s2, 0 for s1 == s2.
  int f = 0;

  // Reads the first string from standard input, including spaces.
  getline(cin, s1);
  // Reads the second string from standard input, including spaces.
  getline(cin, s2);

  /**
   * @brief Loop to compare characters of the two strings one by one.
   *
   * The loop continues as long as `i` is within the bounds of `s1`.
   * It performs case-insensitive comparison by converting characters to uppercase.
   * The loop breaks early if a difference is found.
   * Invariant: `f` correctly reflects the lexicographical order of `s1[0...i-1]` and `s2[0...i-1]`.
   */
  for (int i = 0; i < s1.length(); ++i) {
    ch1 = s1[i];
    ch2 = s2[i];
    // Converts character from s1 to uppercase if it's lowercase.
    if ((int)s1[i] >= 97 && (int)s1[i] <= 122) {
      ch1 = (int)s1[i] - 32;
    }
    // Converts character from s2 to uppercase if it's lowercase.
    if ((int)s2[i] >= 97 && (int)s2[i] <= 122) {
      ch2 = (int)s2[i] - 32;
    }
    // Compares the uppercase versions of the characters.
    if ((int)ch1 < (int)ch2) {
      f = -1; // s1 is lexicographically smaller.
      break;  // No need to compare further.
    } else if ((int)ch1 > (int)ch2) {
      f = 1;  // s1 is lexicographically larger.
      break;  // No need to compare further.
    }
    // If characters are equal, continue to the next pair.
  }

  // @brief Prints the final comparison result (-1, 0, or 1).
  cout << f;

  return 0;
}
