/**
 * @file 71A.CPP
 * @brief Implements a word abbreviation algorithm.
 * * Algorithm: String length-based abbreviation.
 *   If a word's length exceeds 10 characters, it is abbreviated to its first character,
 *   followed by the count of characters between the first and last, and then its last character.
 *   Otherwise, the word remains unchanged.
 * Time Complexity: O(L) for each word, where L is the length of the word. Total O(N*L) for N words.
 * Space Complexity: O(L) for storing each word.
 */

#include <iostream>
using namespace std;

/**
 * @brief Main function to process words based on abbreviation rules.
 * Functional Utility: Handles input/output and orchestrates the word abbreviation logic.
 */
int main() {
  int runs;
  cin >> runs;
  /**
   * Block Logic: Iterates through a series of test cases, processing each word.
   * Invariant: At the start of each iteration, 'run' represents the current test case index.
   */
  for (int run = 0; run < runs; ++run) {
    string word;
    cin >> word;
    /**
     * Block Logic: Applies the word abbreviation rule.
     * Pre-condition: 'word' contains the string to be processed.
     * Invariant: The output string adheres to the abbreviation rule.
     */
    if (word.size() <= 10)
      cout << word << endl;
    else
      cout << word[0] << (word.size() - 2) << word[word.size() - 1] << endl;
  }
  return 0;
}