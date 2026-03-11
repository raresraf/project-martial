/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 *
 * This program reads a number of words and shortens any word that is longer
 * than 10 characters. The shortening rule is to replace the middle characters
 * with the count of those characters, keeping the first and last character.
 *
 * Algorithm: String manipulation based on length.
 * Time Complexity: O(N * L) where N is the number of words and L is the max length of a word.
 *                  Since L is small and constant for string-to-integer conversion,
 *                  it can be considered O(N).
 * Space Complexity: O(L) to store the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the total number of words to process.
  int n;
  // w: stores each word as it's read from input.
  string w;
  cin >> n;
  
  /**
   * @brief Main loop to process each word.
   *
   * Pre-condition: 'n' holds the number of words to be processed.
   * Invariant: At the beginning of each iteration, 'i' words have been processed.
   */
  for (int i = 0; i < n; i++) {
    cin >> w;
    
    /**
     * @brief Conditional block for word shortening.
     *
     * Logic: Checks if the word's length is greater than 10.
     * If true, it prints the abbreviated version. Otherwise, it prints the original word.
     */
    if (w.length() > 10) {
      // Constructs the abbreviated word: first char, length-2, last char.
      cout << w.at(0) << w.length() - 2 << w.back() << endl;
    } else {
      cout << w << endl;
    }
  }
  return 0;
}
