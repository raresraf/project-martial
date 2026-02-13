/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem.
 * @details This program abbreviates words that are longer than 10 characters.
 *
 * @section implementation_note Note on Implementation
 * The loop `for (int i = 0; i <= n; i++)` runs `n+1` times. The first call to
 * `getline(cin, s)` after `cin >> n;` reads the leftover newline character from the
 * input buffer, resulting in an empty string. The code then correctly processes the `n`
 * actual input words on subsequent iterations. Because an empty string has length 0,
 * it is printed as a blank line, which is then followed by the output for the `n` words.
 *
 * @section algorithm Algorithm
 * 1. Read the number of words, `n`.
 * 2. Loop `n+1` times.
 *    a. Read a full line of input into string `s`.
 *    b. Get the length of `s`.
 *    c. If the length is 10 or less, print the string as is.
 *    d. If the length is greater than 10, print the abbreviated form: first character,
 *       length minus 2, and last character.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N * L_max), where N is the number of words and L_max is their maximum length, dominated by I/O operations.
 * - Space Complexity: O(L_max) to store the longest word.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, l;
  string s;
  cin >> n;

  /**
   * @brief This loop processes n+1 lines due to the i <= n condition and the
   * newline character left in the input buffer by `cin >> n;`. The first `getline`
   * call consumes this empty line, which is then printed, and the subsequent n
   * calls read and process the actual words.
   */
  for (int i = 0; i <= n; i++) {
    getline(cin, s);
    l = s.length();
    // Check if the word's length is greater than 10.
    if (l <= 10)
      // If 10 or less, print the word as is. For the first empty line, this prints a blank line.
      cout << s << endl;
    else
      // If greater than 10, print the abbreviation.
      cout << s[0] << l - 2 << s[l - 1] << endl;
  }
}
