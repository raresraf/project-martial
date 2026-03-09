/**
 * @file 71A.CPP
 * @brief Abbreviates words that are longer than 10 characters.
 *
 * This program solves the "Way Too Long Words" problem from Codeforces. It reads a
 * number of words and, for each word, if its length is greater than 10, it prints
 * an abbreviation. The abbreviation consists of the first letter, the number of
 * letters in between, and the last letter. Shorter words are printed unchanged.
 * This version includes I/O optimizations common in competitive programming.
 *
 * @algorithm String Manipulation
 * @complexity
 * - Time: O(N * L), where N is the number of words and L is the maximum length of a word.
 *   This is dominated by the time taken to read the input strings.
 * - Space: O(L) to store the longest word being processed.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Functional Utility: Optimizes I/O by decoupling C++ streams from C stdio
   * and un-tying cin from cout.
   */
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n;
  cin >> n;

  /**
   * @brief Block Logic: Processes each of the `n` words sequentially.
   */
  while (n--) {
    string s;
    cin >> s;

    int t = s.length();

    // Pre-condition: Check if the word's length exceeds the threshold of 10.
    if (t > 10)
      /**
       * @brief Logic: Constructs and prints the abbreviation.
       * Uses `s.at()` for safe character access. The format is the first character,
       * the count of the middle characters (length - 2), and the last character.
       */
      cout << s.at(0) << t - 2 << s.at(t - 1) << "\n";

    else
      // If the word is not too long, print it as is.
      cout << s << "\n";
  }
}