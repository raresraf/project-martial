/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem on Codeforces (71A).
 * @details This program shortens words that are longer than 10 characters.
 * The abbreviation consists of the first character, the number of characters
 * between the first and last, and the last character.
 *
 * @algorithm The program reads an integer `n` for the number of test cases.
 * It then enters a loop that runs `n` times. In each iteration, it reads a word.
 * If the word's length is greater than 10, it constructs and prints the
 * abbreviated version; otherwise, it prints the word as is.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of words to process. This assumes
 * the length of each word is relatively small and bounded, making string
 * operations effectively constant time per word.
 * Space Complexity: O(L), where L is the maximum length of a word, required
 * to store one word at a time.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  /**
   * @brief Block Logic: Processes each of the `n` words.
   * @invariant In each iteration `i`, `i` words have been processed and printed.
   */
  for (int i = 0; i < n; ++i) {
    string s;
    cin >> s;
    /**
     * @brief Block Logic: Checks if the word is "too long" and applies abbreviation rule.
     * Pre-condition: The word `s` has been read from standard input.
     */
    if (s.size() > 10) {
      int a = s.size() - 2;
      // Prints the abbreviated word, e.g., "localization" becomes "l10n".
      printf("%c%d%c\n", s[0], a, s[s.size() - 1]);
    } else
      // If the word is not too long, it is printed unchanged.
      cout << s << endl;
  }

  return 0;
}

// codeforces.com/problemset/problem/71/A