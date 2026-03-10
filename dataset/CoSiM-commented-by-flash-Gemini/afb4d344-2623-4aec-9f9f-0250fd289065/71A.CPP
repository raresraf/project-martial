/**
 * @file 71A.CPP
 * @brief This program solves the "Way Too Long Words" problem (Codeforces 71A).
 *   It abbreviates words longer than 10 characters to their first letter,
 *   the count of intermediate characters, and their last letter. Shorter words are printed as is.
 *   Problem description: https://codeforces.com/problemset/problem/71/A
 * * Algorithm: String length check and conditional formatting.
 *   For words longer than 10 characters, it calculates the number of characters
 *   between the first and last (which is `length - 2`) and formats the word.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 *   The loop to calculate `count` within the abbreviation logic adds a factor of L per word.
 * Space Complexity: O(L_max) to store the longest input word.
 */
// CodeForces
// Way too long words
// GNU C++17

#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.
using namespace std;

/**
 * @brief Main function to process words and apply abbreviation rules.
 * Functional Utility: Reads the number of words, then processes each word
 *                     according to its length, applying abbreviation if necessary.
 */
int main() {
  int numberOfWords; // Stores the total number of words to process.
  // Block Logic: Reads the total number of words.
  cin >> numberOfWords;
  /**
   * Block Logic: Iterates through each word.
   * Pre-condition: `numberOfWords` is a non-negative integer.
   * Invariant: Each iteration processes one word, `numberOfWords` decrements until 0.
   */
  while (numberOfWords > 0) {
    string currentWord; // Stores the current word being processed.
    // Block Logic: Reads the current word.
    cin >> currentWord;
    /**
     * Conditional Logic: Checks if the current word's length is 10 or less.
     * If true, the word is printed as is.
     */
    if (currentWord.length() <= 10) {
      cout << currentWord << endl;
    }
    /**
     * Conditional Logic: If the word's length is greater than 10, it is abbreviated.
     * The abbreviation consists of the first character, the count of intermediate characters,
     * and the last character.
     */
    else {
      int count = 0; // Stores the count of intermediate characters.
      /**
       * Block Logic: Calculates the number of characters between the first and last characters.
       * Inline: This loop effectively calculates `currentWord.length() - 2`.
       *         This could be replaced by a direct arithmetic calculation `count = currentWord.length() - 2;`
       *         for better efficiency, but it achieves the same result.
       */
      for (int i = 1; i < currentWord.length() - 1; i++) {
        count++;
      }
      // Prints the abbreviated word: first char + count + last char.
      cout << currentWord[0] << count << currentWord[currentWord.length() - 1]
           << endl;
    }
    numberOfWords--; // Decrements the word counter.
  }
}
