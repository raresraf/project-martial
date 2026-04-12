/**
 * @file 71A.CPP
 * @brief Implements a word abbreviation algorithm for competitive programming.
 *   If a word's length exceeds 10 characters, it is abbreviated to its first character,
 *   followed by the count of characters between the first and last, and then its last character.
 *   Otherwise, the word remains unchanged.
 * * Algorithm: String Length-Based Abbreviation.
 *   Reads N words. For each word, it checks its length. If longer than 10, it constructs
 *   an abbreviated version. Otherwise, the original word is printed.
 * Time Complexity: O(L) for each word, where L is the length of the word. Total O(N*L) for N words.
 * Space Complexity: O(L) for storing each word.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to process words based on abbreviation rules.
 * Functional Utility: Handles input/output and orchestrates the word abbreviation logic for multiple test cases.
 */
int main() {
  int n, crlen; // 'n' is the number of words to process, 'crlen' stores the current word's length.
  string cr;   // 'cr' stores the current word.

  cin >> n; // Reads the total number of words (test cases) to follow.
  // Consumes the remaining newline character after reading 'n'.
  // This is crucial when mixing cin >> int and getline(cin, string).
  getline(cin, cr); 

  /**
   * Block Logic: Iterates through each word (test case) provided in the input.
   * Invariant: At the start of each iteration, 'i' represents the current word index.
   */
  for (int i = 0; i <= n; i++) { // Loop runs n+1 times because of the initial getline.
    getline(cin, cr);            // Reads an entire line (word) from input.
    crlen = cr.length();         // Gets the length of the current word.

    /**
     * Block Logic: Applies the word abbreviation rule based on length.
     * Pre-condition: 'cr' contains the string to be processed and 'crlen' is its length.
     * Invariant: The output string adheres to the abbreviation rule.
     */
    if (crlen > 10) {
      // Prints the abbreviated word: first char + (length-2) + last char.
      cout << cr[0] << crlen - 2 << cr[crlen - 1] << endl;
      continue; // Moves to the next iteration after printing.
    }
    cout << cr << endl; // Prints the original word if its length is 10 or less.
  }
  return 0; // Indicates successful program execution.
}
