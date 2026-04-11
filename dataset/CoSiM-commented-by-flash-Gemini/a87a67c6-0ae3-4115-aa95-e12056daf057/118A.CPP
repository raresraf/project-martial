/**
 * @file 118A.CPP
 * @brief This program takes a word as input and transforms it by converting
 * all uppercase letters to lowercase, removing vowels ('a', 'e', 'i', 'o', 'u', 'y'),
 * and prepending a dot ('.') before each remaining consonant.
 *
 * Algorithm: String iteration and character-by-character transformation.
 * Time Complexity: O(L) where L is the length of the input word.
 * Space Complexity: O(L) to store the input word.
 */

#include <iostream>
#include <string>
#include <cctype> // Required for tolower

using namespace std;

int main() {
  // Functional Utility: Declares a string 'word' to store the input word.
  string word;
  // Block Logic: Reads a single word from standard input.
  cin >> word;

  /**
   * Block Logic: Iterates through each character of the input word.
   * Invariant: 'c' holds the current character being processed.
   */
  for (char c : word) {
    // Functional Utility: Converts the current character to lowercase.
    c = tolower(c);
    /**
     * Block Logic: Checks if the current character is a consonant (not a vowel).
     * Pre-condition: 'c' is the lowercase version of the original character.
     * Invariant: Only consonants will be processed further.
     */
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')
      // Functional Utility: Prints a dot followed by the consonant.
      cout << "." << c;
  }
  return 0;
}
