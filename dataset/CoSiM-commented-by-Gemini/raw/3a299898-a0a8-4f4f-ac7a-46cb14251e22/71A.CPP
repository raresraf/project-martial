/**
 * @file 71A.CPP
 * @brief A standard C++ solution to the "Way Too Long Words" problem (Codeforces 71A).
 *
 * @details This program abbreviates words that are longer than 10 characters.
 * The abbreviation is formed by concatenating the first character, the count of
 * the intermediate characters, and the last character.
 *
 * @algorithm The program reads an integer `n`, the number of words to process.
 * It then loops `n` times, reading one word per iteration. For each word, it
 * checks if its length is greater than 10. If it is, the abbreviated form is
 * printed. Otherwise, the word is printed as is.
 *
 * @complexity
 * Time Complexity: O(N * L), where N is the number of words and L is their
 *                  maximum length. The complexity is dominated by I/O operations.
 * Space Complexity: O(L), to store the longest word read from input.
 */
#include <iostream>

using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from C streams for faster I/O.
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;

  /**
   * @brief Block Logic: Processes each of the `n` words.
   * @invariant In each iteration `i`, `i` words have been processed and their
   * (potentially abbreviated) form has been printed.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;

    int len = s.length();

    // Check if the word's length exceeds the threshold for abbreviation.
    if (len > 10) {
      cout << s[0] << len - 2 << s[len - 1] << endl;
    } else {
      cout << s << endl;
    }
  }

  return 0;
}
