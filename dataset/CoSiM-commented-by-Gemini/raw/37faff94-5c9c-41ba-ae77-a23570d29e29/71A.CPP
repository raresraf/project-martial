/**
 * @file 71A.CPP
 * @brief Abbreviates long words according to specific rules.
 * @see https://codeforces.com/problemset/problem/71/A
 *
 * This program processes a list of words. For each word, if its length is
 * greater than 10 characters, it is replaced by an abbreviation. The
 * abbreviation consists of the first character of the word, followed by the
 * number of characters between the first and last characters, and finally the
 * last character of the word. Words with 10 or fewer characters are printed
 * as they are.
 *
 * Algorithm:
 * 1. Read an integer `n`, the number of words to process.
 * 2. Loop `n` times:
 *    a. Read the word into a string `s`.
 *    b. Check if the length of `s` is greater than 10.
 *    c. If it is, construct the abbreviation: `s[0]`, `s.size() - 2`, `s[s.size() - 1]`.
 *    d. If not, print the string `s` as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum length
 * of a word. The processing for each word is constant time, but reading the string
 * takes time proportional to its length.
 * Space Complexity: O(L), where L is the maximum length of a word, to store the string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  // Read the number of words to be processed.
  cin >> n;

  /**
   * @brief Main processing loop that iterates `n` times.
   * In each iteration, a word is read and processed based on its length.
   */
  for (int i = 0; i < n; ++i) {
    string s;
    cin >> s;
    /**
     * @brief Pre-condition: A word `s` has been read.
     * Block Logic: Checks if the word is "too long" (more than 10 characters).
     * If so, it prints an abbreviation; otherwise, it prints the word itself.
     */
    if (s.size() > 10) {
      // Calculate the number of internal characters.
      int a = s.size() - 2;
      // Print the abbreviation in the format: first char, count, last char.
      printf("%c%d%c\n", s[0], a, s[s.size() - 1]);
    } else {
      // If the word is not too long, print it as is.
      cout << s << endl;
    }
  }

  return 0;
}

// codeforces.com/problemset/problem/71/A