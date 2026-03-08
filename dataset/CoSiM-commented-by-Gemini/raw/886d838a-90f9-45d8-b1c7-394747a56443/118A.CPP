/**
 * @file 118A.CPP
 * @brief Solution for a string processing task. This program transforms a given
 * string by deleting all vowels, inserting a character "." before each
 * consonant, and converting all uppercase consonants to corresponding
 * lowercase ones.
 *
 * @details
 * The program reads a string, iterates through each character, and applies the
 * following rules:
 * 1. Convert the character to lowercase to handle both cases uniformly.
 * 2. Check if the character is a vowel ('a', 'e', 'i', 'o', 'u', 'y').
 * 3. If it is a vowel, it is ignored and effectively deleted.
 * 4. If it is a consonant, a '.' is printed, followed by the consonant itself.
 *
 * @section
 * Algorithm: Single-pass string transformation.
 * Time Complexity: O(N), where N is the length of the input string, as it involves a single loop through the string.
 * Space Complexity: O(N), to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int i;
  char s[100]; // Buffer to hold the input string.

  // Read the input string from standard input.
  scanf("%s", s);

  int x = strlen(s); // Get the length of the string.

  /**
   * @brief Main processing loop that iterates through each character of the string.
   *
   * Invariant: Processes characters one by one from left to right.
   */
  for (i = 0; i < x; i++) {
    // Convert the current character to lowercase for case-insensitive comparison.
    s[i] = tolower(s[i]);

    /**
     * @brief Conditional check to distinguish between vowels and consonants.
     * The character 'y' is treated as a vowel in this context.
     *
     * Pre-condition: The character s[i] is in lowercase.
     */
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      // If the character is a vowel, skip it and continue to the next iteration.
      continue;
    } else {
      // If the character is a consonant, print a '.' followed by the character.
      cout << "." << s[i];
    }
  }
  // The original commented out lines are preserved.
  // cout<<s;
}
