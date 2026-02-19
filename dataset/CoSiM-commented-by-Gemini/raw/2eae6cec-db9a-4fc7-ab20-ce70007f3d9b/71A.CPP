/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem on Codeforces.
 * @details This program shortens words that are longer than 10 characters by replacing the middle
 * characters with a count of their number.
 *
 * Algorithm:
 * 1. Read an integer `n`, which is the number of words to be processed.
 * 2. Loop `n` times.
 * 3. In each iteration, read a single word as a string `s`.
 * 4. Check if the length of the string `s` is greater than 10.
 * 5. If it is, the word is abbreviated:
 *    - Print the first character of the string.
 *    - Print the number of characters between the first and last characters, which is `length - 2`.
 *    - Print the last character of the string.
 * 6. If the length is 10 or less, print the string as is.
 *
 * @note There is a subtle bug in the original code: `cout << s[i] ...`. The variable `i` is declared
 * outside the `while` loop and never modified within it, so it always remains 0. The code works by
 * coincidence because the first character's index is indeed 0. The correct, non-ambiguous way
 * would be to use `s[0]` or `s.front()`.
 *
 * Time Complexity: O(n * L), where `n` is the number of words and `L` is the maximum length of a word, due to string I/O.
 * The processing for each word is O(L).
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  int i = 0; // This variable is used in a potentially buggy way inside the loop.
  cin >> n;
  string s;

  // Loop `n` times to process each word.
  while (n--) {
    cin >> s;

    /**
     * Block Logic: Check if the word is "too long" and abbreviate it if necessary.
     */
    if (s.length() > 10) {
      // Abbreviate the word. Note the use of `s[i]` which happens to work because `i` is 0.
      cout << s[i] << s.length() - 2 << s[s.length() - 1] << "\n";
    } else {
      // If the word is not too long, print it as is.
      cout << s << endl;
    }
  }

  return 0;
}
