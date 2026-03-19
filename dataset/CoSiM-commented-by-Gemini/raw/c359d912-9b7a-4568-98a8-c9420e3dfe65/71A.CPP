/*

Problem description:

https://codeforces.com/problemset/problem/71/A

*/
/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem (Codeforces 71A).
 * @details This program shortens words that are longer than 10 characters.
 * The abbreviation consists of the first letter, the count of letters
 * between the first and last, and the last letter.
 *
 * Algorithm: Conditional Abbreviation
 * The program reads a number 'n', indicating how many words to process. It then
 * loops 'n' times, reading each word. For each word, it checks its length. If the
 * length is greater than 10, it prints the special abbreviation. Otherwise,
 * it prints the word unchanged.
 *
 * NOTE: The loop `for (int i = 0; i <= n; i++)` combined with `getline` after
 * `cin >> n` is a common pitfall. The first `getline` consumes the newline
 * character left in the input buffer by `cin >> n`, resulting in an extra loop
 * iteration that processes an empty string. The subsequent 'n' iterations correctly
 * process the intended words.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum
 * length of a word, dominated by I/O operations.
 * Space Complexity: O(L) to store the longest word.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // n: the number of words to process.
  int n;
  // crlen: the length of the current word.
  int crlen;
  // cr: the current word being processed.
  string cr;
  cin >> n;

  /**
   * Block Logic: Loop to read and process n+1 lines.
   * WARNING: This loop runs n+1 times. The first iteration after `cin >> n`
   * will read an empty string because of the leftover newline in the input buffer.
   * The following 'n' iterations will correctly read the words.
   */
  for (int i = 0; i <= n; i++) {
    getline(cin, cr);
    crlen = cr.length();
    
    // Core Logic: Check if the word is "too long".
    if (crlen > 10) {
      // If so, print the abbreviation: first char, length-2, last char.
      cout << cr[0] << crlen - 2 << cr[crlen - 1] << endl;
      // This continue is redundant but harmless.
      continue;
    }
    
    // For words 10 characters or shorter (and the initial empty string), print them as is.
    cout << cr << endl;
  }
  return 0;
}