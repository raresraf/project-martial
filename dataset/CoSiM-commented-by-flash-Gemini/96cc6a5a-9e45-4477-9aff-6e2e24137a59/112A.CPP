/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two input strings using a custom lowercase conversion.
 * * Algorithm: Character-by-Character Lexicographical Comparison with Custom Lowercase Conversion.
 * Time Complexity: $O(L)$ where L is the length of the shorter string, due to early exit on difference.
 * Space Complexity: $O(1)$ additional space, as transformations are in-place.
 *
 * This program reads two strings, converts them to lowercase using a helper function,
 * and then compares them character by character lexicographically. It prints 1 if the
 * first string is greater, -1 if the first is less, and 0 if they are equal.
 */

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

#define ll long long // Defines a macro for long long integer type (unused in this specific problem).

/**
 * @brief Converts an uppercase character to its lowercase equivalent.
 *
 * This helper function checks if the input character `s` is an uppercase letter.
 * If it is, it adds 32 (the ASCII difference between uppercase and lowercase letters)
 * to convert it to lowercase. Otherwise, it returns the character unchanged.
 *
 * @param s The character to potentially convert.
 * @return The lowercase version of `s` if it was uppercase, otherwise `s` itself.
 */
char down(char s) { return (s <= 'Z' ? s + 32 : s); }

int main() {
  string s1, s2; // Declares two strings to store input.
  cin >> s1 >> s2; // Reads the two input strings.

  /**
   * Block Logic: Converts the first input string `s1` to lowercase in-place.
   * Pre-condition: `s1` contains an input string.
   * Invariant: After the loop, `s1` will contain only lowercase characters.
   */
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = down(s1[i]); // Converts each character of `s1` to lowercase.
  }
  
  /**
   * Block Logic: Converts the second input string `s2` to lowercase in-place.
   * Pre-condition: `s2` contains an input string.
   * Invariant: After the loop, `s2` will contain only lowercase characters.
   */
  for (int i = 0; i < s1.length(); i++) { // Note: Loop condition `s1.length()` is used for `s2`. This assumes `s1` and `s2` have similar lengths or `s1` is not shorter than `s2`.
    s2[i] = down(s2[i]); // Converts each character of `s2` to lowercase.
  }
  
  /**
   * Block Logic: Performs a character-by-character lexicographical comparison of the two lowercase strings.
   * Pre-condition: Both `s1` and `s2` contain only lowercase characters.
   * Invariant: The loop continues as long as characters at corresponding positions are equal.
   * Post-condition: Prints 1, -1, or 0 based on the first differing character or if strings are equal.
   */
  for (int i = 0; i < s1.length(); i++) { // Note: Loop condition `s1.length()` is used for comparison. This assumes `s1` and `s2` have similar lengths or `s1` is not shorter than `s2`.
    if (s1[i] > s2[i]) {
      cout << 1 << "\n"; // Inline: `s1` is lexicographically greater than `s2`.
      return 0;           // Exits early as difference is found.
    } else if (s1[i] < s2[i]) {
      cout << -1 << "\n"; // Inline: `s1` is lexicographically less than `s2`.
      return 0;           // Exits early as difference is found.
    }
  }
  // Block Logic: If the loop completes without finding any differences, the strings are equal.
  cout << 0 << "\n"; // Inline: `s1` is lexicographically equal to `s2`.
  return 0;          // Standard successful exit code for main function.
}
