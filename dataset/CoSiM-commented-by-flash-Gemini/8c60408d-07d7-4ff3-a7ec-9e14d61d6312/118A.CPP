/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: String processing with character-by-character filtering and transformation.
 *   It iterates through the input string, checking if each character is a vowel (case-insensitively).
 *   If it's a consonant, it prepends a '.' to a new string and appends the lowercase version of the consonant.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass through the string.
 * Space Complexity: O(L) to store the processed characters in the `result` string. In the worst case (all consonants), the output string length can be up to `2*L`.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <string>, <cctype>.
using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Reads the input string, applies the transformation rules (vowel removal,
 *                     consonant prepending with '.', and lowercase conversion), and
 *                     builds and prints the resulting string.
 */
int main() {
  string word, result; // `word` stores the original input string, `result` builds the transformed string.
  // Block Logic: Reads the input string `word`.
  cin >> word;
  /**
   * Block Logic: Iterates through each character of the input string `word`.
   * Pre-condition: `word` is populated with the input string.
   * Invariant: `result` is progressively built with transformed characters.
   */
  for (int i = 0; i < word.length(); i++) {
    /**
     * Conditional Logic: Checks if the current character `word[i]` is a vowel (both lowercase and uppercase).
     * If it is a vowel, the `continue` statement skips to the next iteration, effectively deleting the vowel.
     */
    if (word[i] == 'a' || word[i] == 'A' || word[i] == 'e' || word[i] == 'E' ||
        word[i] == 'i' || word[i] == 'I' || word[i] == 'o' || word[i] == 'O' ||
        word[i] == 'u' || word[i] == 'U' || word[i] == 'y' || word[i] == 'Y')
      continue; // Skip vowels.
    else {
      // If the character is a consonant (not a vowel), prepend a '.' to `result`.
      result += '.';
      // Then append the lowercase version of the consonant to `result`.
      // Inline: `tolower(word[i])` converts the character to its lowercase equivalent.
      result += tolower(word[i]);
    }
  }
  // Block Logic: Prints the final transformed string followed by a newline.
  cout << result << endl;

  return 0;
}
