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
  // Functional Utility: Initializes `n` for the number of words.
  int n;
  // Block Logic: Reads the total number of words (N) to be processed.
  cin >> n;

  /**
   * Block Logic: Iterates N times, processing one word in each iteration.
   * Invariant: In each iteration, `a` holds the current word to be processed.
   *            The loop continues until `n` words have been read and processed.
   */
  for (int i = 0; i < n; i++) {
    string a;
    // Block Logic: Reads the current word into string `a`.
    cin >> a;

    // Pre-condition: Checks if the word's length is 10 characters or shorter.
    if (a.length() <= 10) {
      // Block Logic: If the word is short, prints it as is, followed by a newline.
      cout << a << endl;
      ; // Empty statement, no functional effect.
    } else {
      // Block Logic: If the word is too long, prints its abbreviation.
      // Format: first character + (length - 2) + last character, followed by a newline.
      cout << a[0] << a.length() - 2 << a[a.length() - 1] << endl;
    }
  }
  return 0;
}
