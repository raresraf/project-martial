/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and formatting consonants.
 *
 * This program takes a string as input and applies the following rules:
 * 1. All uppercase letters are converted to their lowercase equivalents.
 * 2. All vowels ('a', 'e', 'i', 'o', 'u', 'y') are removed.
 * 3. A period '.' is inserted before each consonant.
 * The resulting string is printed to the standard output.
 *
 * Algorithm:
 * 1. Read the input string into a character array `s`.
 * 2. Iterate through the string character by character.
 * 3. In each iteration, convert the current character to lowercase.
 * 4. Check if the lowercase character is a vowel. The vowels are considered
 *    to be 'a', 'e', 'i', 'o', 'u', and 'y'.
 * 5. If the character is a vowel, it is ignored, and the loop continues.
 * 6. If the character is a consonant, a '.' is printed, followed by the character.
 * 7. After the loop, print a newline character to complete the output.
 *
 * Time Complexity: O(N), where N is the length of the input string, as the
 * program performs a single pass over the string.
 * Space Complexity: O(N) to store the input string `s`.
 */
#include <ctype.h>
#include <iostream>

using namespace std;

int main() {
  char s[101];
  char aux; // aux is an unused variable.
  int i = 0;

  // Read the input string.
  cin >> s;

  /**
   * @brief Loop through the string to process each character.
   * Pre-condition: `s` contains the input string.
   */
  while (s[i]) {
    // Convert the character to lowercase for case-insensitive vowel checking.
    s[i] = tolower(s[i]);

    /**
     * @brief Check if the character is a vowel.
     * Vowels are 'a', 'e', 'i', 'o', 'u', 'y'.
     */
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      // If it's a vowel, skip it and move to the next character.
      i++;
      continue;
    } else {
      // If it's a consonant, print a dot followed by the character.
      cout << "." << s[i];
      i++;
    }
  }

  cout << endl;

  return 0;
}