/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem. This program reads a
 *        list of words and abbreviates any word longer than 10 characters.
 *        Abbreviation consists of the first letter, the number of characters
 *        between the first and last letters, and the last letter.
 *        Words 10 characters or shorter are printed as they are.
 *
 * Algorithm: String length check and conditional formatting.
 *            For each input word, its length is checked. If it exceeds 10,
 *            it's formatted as `first_char + (length - 2) + last_char`.
 *            Otherwise, the original word is printed.
 * Time Complexity: O(N * L), where N is the number of words and L is the
 *                  average length of a word. In the worst case, if all words
 *                  are long, each character is processed once.
 * Space Complexity: O(L) for storing the current word.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Initializes variables for loop control and string manipulation.
  // `n` for the number of words, `i` for loop counter, `k` for string size.
  long long int n;
  // Block Logic: Reads the total number of words (N) to be processed.
  cin >> n;
  string a;
  /**
   * Block Logic: Iterates N times, processing one word in each iteration.
   * Invariant: In each iteration, `a` holds the current word to be processed.
   *            The loop continues until `n` words have been read and processed.
   */
  for (long long int i = 0; i < n; i++) {
    // Block Logic: Reads the current word into string `a`.
    cin >> a;
    // Functional Utility: Calculates the length of the current word.
    long long int k = a.size();
    // Pre-condition: Checks if the word's length is greater than 10 characters.
    if (k > 10) {
      // Block Logic: If the word is too long, prints its abbreviation.
      // Format: first character + (length - 2) + last character.
      cout << a[0] << k - 2 << a[k - 1];
    } else {
      // Block Logic: If the word is 10 characters or shorter, prints it as is.
      cout << a;
    }
    // Block Logic: Prints a newline character after each word (or abbreviation)
    // for proper formatting of output.
    cout << "\n";
  }
  return 0;
}