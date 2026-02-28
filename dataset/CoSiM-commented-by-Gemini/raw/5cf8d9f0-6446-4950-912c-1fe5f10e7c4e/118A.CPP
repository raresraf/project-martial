/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem from Codeforces.
 * @details The problem requires processing a string by deleting vowels, inserting a '.'
 * before each consonant, and converting all consonants to lowercase. This implementation
 * attempts to solve it using a two-pass approach: first replacing vowels with a '0'
 * placeholder, and then processing the remaining characters.
 *
 * @algorithm Flawed Two-Pass String Manipulation
 * 1. First Pass (Vowel Marking): Iterate through the input string. For each character,
 *    check if it's a vowel. If so, replace it with a '0'.
 *    **Note: The vowel-checking logic in this implementation is buggy.**
 * 2. Second Pass (Consonant Processing): Iterate through the modified string. If a
 *    character is not '0', prepend a '.' to it, convert it to lowercase, and print it.
 *
 * @complexity
 * - Time: O(N), where N is the length of the string. The code iterates through the
 *   string twice.
 * - Space: O(N), to store the input string.
 */
#include <bits/stdc++.h>
#include <cstring>
using namespace std;

int main() {
  // s: the input string.
  // vowels: a string containing all vowels for easy lookup.
  // rp: the character used to replace vowels, acting as a placeholder.
  string s, vowels = "aoyeui", rp = "0";
  cin >> s;
  int n = s.length();

  /**
   * @brief First pass: Mark all vowels by replacing them with '0'.
   * This loop intends to find all vowels (case-insensitively) and replace
   * them with a placeholder '0' in the original string.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief BUGGY Vowel Check
     * This condition is incorrect. `string::find` returns `string::npos` (a large
     * unsigned value) when a character is not found. Comparing this with signed
     * integers like `0` leads to unexpected behavior. A correct check would be:
     * `vowels.find(tolower(s[i])) != string::npos`
     */
    if (vowels.find(tolower(s[i])) >= 0 && vowels.find(tolower(s[i])) <= n) {
      s.replace(i, 1, rp);
    }
  }

  /**
   * @brief Second pass: Process consonants and print the result.
   * This loop iterates through the modified string. For any character that is
   * not the '0' placeholder, it prepends a '.' and prints the character in lowercase.
   */
  for (int i = 0; i < n; i++) {
    // Pre-condition: Process the character only if it's not a marked vowel.
    if (s[i] != '0') {
      // Check for uppercase characters using their ASCII values (A=65, Z=90).
      if (s[i] > 64 && s[i] < 91) {
        // If uppercase, print a '.' and the character converted to lowercase.
        // Adding 32 to an uppercase ASCII value converts it to lowercase.
        cout << "." << (char)(s[i] + 32);
      } else {
        // If already lowercase, just print a '.' and the character.
        cout << "." << s[i];
      }
    }
  }
  return 0; // Indicate successful execution.
}
