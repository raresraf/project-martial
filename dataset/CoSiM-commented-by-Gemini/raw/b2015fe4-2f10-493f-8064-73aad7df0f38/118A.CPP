/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 * @details This program processes a string by removing vowels, prepending consonants with a '.',
 * and converting all consonants to lowercase.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. In each iteration, first convert the character to its lowercase equivalent. This implementation
 *    does so by checking its ASCII value against 90 (for 'Z') and adding 32 if it's uppercase.
 * 4. Check if the now-lowercase character is one of the vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 5. If it's a vowel, it is ignored and the loop continues.
 * 6. If it's a consonant, a '.' is printed, followed by the lowercase consonant.
 *
 * @complexity
 * - Time: O(L), where L is the length of the string, due to a single pass over the string.
 * - Space: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // Unused variable.
  string s;
  cin >> s;

  /**
   * @brief Main loop to process each character of the string.
   */
  for (int i = 0; i < s.length(); i++) {
    // Inline: Converts character to lowercase using ASCII arithmetic.
    // Checks if the ASCII value is <= 90 (for 'Z') and adds 32 if it's an uppercase letter.
    if (s[i] <= 90)
      s[i] += 32;

    // Pre-condition: Check if the character, now lowercase, is a vowel.
    // The problem defines 'y' as a vowel.
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      // Action: If it's a vowel, do nothing and continue to the next character.
      continue;
    } else
      // Action: If it's a consonant, print a '.' followed by the lowercase character.
      cout << "." << s[i];
  }
}