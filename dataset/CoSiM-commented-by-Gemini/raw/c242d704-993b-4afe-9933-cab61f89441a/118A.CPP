/**
 * @file 118A.CPP
 * @brief C-style solution for the "String Task" problem (Codeforces 118A).
 * @details This program transforms an input string based on a set of rules:
 * 1. Converts all characters to lowercase.
 * 2. Removes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 3. Inserts a '.' character before every consonant.
 *
 * This implementation uses C-style strings and functions (`char[]`, `scanf`, `strlen`, `printf`).
 *
 * Algorithm: Single-Pass Character Transformation
 * The program iterates through the string character by character. In each iteration, it first
 * converts any uppercase character to its lowercase equivalent. Then, it checks if the
 * character is a vowel. If it's a consonant, it prints a '.' followed by the character.
 *
 * NOTE: The use of `scanf("%s", &s)` with a fixed-size buffer is unsafe and can lead to
 * buffer overflows if the input string exceeds the buffer's capacity. Additionally, calling
 * `strlen()` in the loop's condition is inefficient as it recomputes the string length on
 * every iteration.
 *
 * Time Complexity: O(L^2) in the worst case because of the `strlen` call inside the loop,
 * where L is the string length. A standards-compliant compiler cannot optimize away the `strlen`
 * call as the string is modified within the loop.
 * Space Complexity: O(100) due to the fixed-size character array.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  int i;
  // A fixed-size C-style character array to store the string.
  // WARNING: This is vulnerable to buffer overflows if input is > 99 chars.
  char s[100];
  // Reads a word from standard input into the buffer.
  scanf("%s", &s);

  /**
   * Block Logic: Iterate over each character of the input string to process it.
   * Note: `strlen(s)` is called on each iteration, which is inefficient.
   * A better approach would be to calculate the length once before the loop.
   */
  for (i = 0; i < strlen(s); i++) {
    // Block Logic: Convert the character to lowercase if it's uppercase.
    // This relies on ASCII character codes.
    if (s[i] >= 65 && s[i] <= 90) {
      s[i] = 97 + s[i] - 65;
    }

    // Block Logic: Check if the character is a consonant.
    // Pre-condition: The character s[i] is now lowercase.
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y') {
      // If it's a consonant, print a '.' followed by the character.
      printf(".%c", s[i]);
    }
  }
  return 0;
}