/**
 * @file 71A.CPP
 * @brief Logic for automated string abbreviation based on length thresholds.
 * 
 * Logic: Implements a numeronym generation algorithm (e.g., "l10n" for "localization"). 
 * Words longer than 10 characters are compressed into a format: 
 * (first character + count of middle characters + last character).
 * 
 * Time Complexity: $O(N \times L)$ where N is the number of test cases and L is the string length.
 * Space Complexity: $O(L)$ to store the processing string.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  int t;
  string word;
  
  // Functional Utility: Reads the total number of independent words to process.
  cin >> t;

  /**
   * Block Logic: Iterative processing of input words.
   * Invariant: Each word is processed independently, preserving state only for the 
   * duration of the current loop iteration.
   */
  while (t--) {
    cin >> word;

    /**
     * Functional Utility: Conditional compression logic for "too long" words.
     * Logic: Threshold for compression is strictly > 10 characters.
     */
    if (word.length() > 10) {
      string new_word;
      // Extracts boundary characters using string constructors.
      string first(1, word[0]);
      string last(1, word[word.length() - 1]);
      
      /**
       * Functional Utility: Assembles the numeronym.
       * Logic: Concatenates the first letter, the numeric count of interior characters, 
       * and the final letter.
       */
      new_word = first + to_string(word.length() - 2) + last;
      cout << new_word << endl;
    } else {
      // Logic: Preserves the original string for entities below the length threshold.
      cout << word << endl;
    }
  }
  
  return 0;
}
