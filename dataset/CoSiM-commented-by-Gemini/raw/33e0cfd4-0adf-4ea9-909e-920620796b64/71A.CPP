/**
 * @file 71A.CPP
 * @brief Abbreviates long words.
 *
 * This program implements a solution for a common competitive programming problem
 * known as "Way Too Long Words". It reads a list of words and shortens any word
 * that is longer than 10 characters. The abbreviation consists of the first letter,
 * the number of letters between the first and last, and the last letter.
 * For example, "localization" becomes "l10n".
 *
 * Algorithm:
 * 1. Read the number of test cases (t).
 * 2. Loop t times:
 *    a. Read a word (string s).
 *    b. Check if the length of s is greater than 10.
 *    c. If it is, print the abbreviated version.
 *    d. Otherwise, print the word as is.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the maximum length of a word.
 *                  Since L is usually small and constant in such problems, it can be considered O(T).
 * Space Complexity: O(L) to store the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes
   * to optimize high-throughput I/O required for competitive programming constraints.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int t;
  // Read the number of words to process.
  cin >> t;
  /**
   * Block Logic: Main processing loop.
   * Iterates for each test case.
   */
  while (t--) {
    string s;
    cin >> s;
    /**
     * Block Logic: Abbreviation rule.
     * Pre-condition: A word `s` has been read.
     * If the word's length is 10 or less, it's printed verbatim.
     * Otherwise, it's abbreviated to the first character, the count of
     * intermediate characters (length - 2), and the last character.
     */
    (s.length() <= 10)
        ? cout << s << "\n"
        : cout << s[0] << (s.length() - 2) << s[s.length() - 1] << "\n";
  }
}
