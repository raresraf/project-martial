/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - "Way Too Long Words".
 * @details The program shortens words that are longer than 10 characters into a special
 * abbreviation. The abbreviation consists of the first character, the number of
 * characters between the first and last, and the last character.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the number of words to process, `runs`.
 * 2. Loop `runs` times to process each word.
 * 3. In each iteration, read the input `word`.
 * 4. Check the length of the word.
 * 5. If the word's length is 10 or less, print the word unchanged.
 * 6. If the word's length is greater than 10, construct and print the abbreviation:
 *    - The first character: `word[0]`
 *    - The count of middle characters: `word.size() - 2`
 *    - The last character: `word[word.size() - 1]`
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of words, as each word is processed in constant time
 * (string length calculation and printing are very fast for typical constraints).
 * Space Complexity: O(L_max), where L_max is the length of the longest word.
 */
#include <iostream>
using namespace std;
int main() {
  int runs;
  // Read the number of test cases (words to process).
  cin >> runs;
  /**
   * Block Logic: This loop processes each of the 'runs' words.
   */
  for (int run = 0; run < runs; ++run) {
    string word;
    cin >> word;
    /**
     * Block Logic: Checks if the word is "too long" (more than 10 characters).
     * If it is, the word is abbreviated. Otherwise, it is printed as is.
     */
    if (word.size() <= 10)
      cout << word << endl;
    else
      // Construct the abbreviation: first char, (length - 2), last char.
      cout << word[0] << (word.size() - 2) << word[word.size() - 1] << endl;
  }
}