/**
 * @file 118A.CPP
 * @brief Transforms a given word by filtering out vowels and prepending '.' to consonants.
 *   This program implements a string transformation rule: convert all letters to lowercase,
 *   then remove all vowels (a, o, y, e, u, i). Finally, prepend each consonant with a '.'.
 * * Algorithm: Character-by-Character Transformation.
 *   Iterates through each character of the input word, converts it to lowercase,
 *   and checks if it's a vowel. If it's a consonant, it's appended to the result
 *   string with a preceding dot.
 * Time Complexity: O(L) where L is the length of the input word.
 * Space Complexity: O(L) for storing the input word and the result string.
 */

#include <iostream>
#include <string> // Required for std::string
#include <cctype> // Required for tolower

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Handles input, applies the transformation rules, and prints the result.
 */
int main() {
  std::string word;   // The input word.
  std::string result; // The transformed result string.
  std::cin >> word;   // Reads the input word.
  char c;             // Temporary variable to hold the lowercase character.

  /**
   * Block Logic: Iterates through each character of the input word.
   * Invariant: 'i' is the current character index. 'c' holds the lowercase version of 'word[i]'.
   */
  for (int i = 0; i < word.length(); i++) {
    c = std::tolower(word[i]); // Converts the current character to lowercase.
    /**
     * Block Logic: Checks if the lowercase character is a vowel.
     * If it is a vowel (a, i, u, e, o, y), the character is skipped.
     * If it's a consonant, a '.' is appended to 'result' followed by the consonant.
     */
    switch (c) {
    case 'a':
    case 'i':
    case 'u':
    case 'e':
    case 'o':
    case 'y':
      continue; // Skip vowels.
    }
    result.append(1, '.'); // Appends a dot.
    result.append(1, c);   // Appends the consonant.
  }
  std::cout << result << std::endl; // Prints the final transformed string.
  return 0; // Indicates successful program execution.
}