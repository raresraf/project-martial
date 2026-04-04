/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem on platforms like Codeforces.
 * @details This program processes a series of words. If a word is longer than 10 characters,
 * it is replaced by a special abbreviation. Otherwise, it is printed as is.
 *
 * Algorithm:
 * 1. Read the number of test cases, t.
 * 2. Loop t times:
 *    a. Read a word.
 *    b. Check if the length of the word is greater than 10.
 *    c. If it is, print the first character, the number of characters between the first and last (length - 2), and the last character.
 *    d. Otherwise, print the word unchanged.
 *
 * Time Complexity: O(T * L) where T is the number of test cases and L is the maximum length of a word.
 * Since L is small for typical inputs, it can be considered O(T).
 * Space Complexity: O(L) for storing each word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, i;
  cin >> t;
  /**
   * @brief Main loop to process each test case.
   * Pre-condition: 't' contains the number of words to process.
   * Invariant: The loop will execute 't' times.
   */
  while (t--) {
    string a;
    cin >> a;
    /**
     * @brief Conditional block to check word length.
     * Logic: Implements the problem's core requirement. If a word's length
     * exceeds 10, it's abbreviated. Otherwise, it's printed verbatim.
     */
    if (a.size() > 10) {
      // Abbreviate the word. The format is:
      // [first character][number of characters between first and last][last character]
      cout << a[0] << a.size() - 2 << a[a.size() - 1] << endl;
    } else {
      // The word is not too long, so print it as it is.
      for (i = 0; i < a.size(); i++) {
        cout << a[i];
      }
      cout << endl;
    }
  }
}