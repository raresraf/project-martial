/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem (112A) from Codeforces.
 * @details This program compares two strings lexicographically, ignoring differences in
 * letter case. This implementation uses an unconventional character-by-character
 * reading and comparison approach.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution reads the first string entirely, then reads the second string
 * character by character, performing an immediate case-insensitive comparison at each step.
 * 1. Read the first string `s1`.
 * 2. Loop from the first to the last character of `s1`.
 * 3. In each iteration, read a single character `c` from the input stream, which will be
 *    the corresponding character from the second string.
 * 4. Convert both `s1[i]` and `c` to lowercase for a case-insensitive comparison.
 * 5. If `s1[i]` is less than `c`, the first string is lexicographically smaller. Print -1
 *    and terminate.
 * 6. If `s1[i]` is greater than `c`, the first string is larger. Print 1 and terminate.
 * 7. If the loop completes without any differences, the strings are equal. Print 0.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings, due to the single pass.
 * Space Complexity: O(L), to store the first string.
 */
#include <bits/stdc++.h>
using namespace std;

#define DBG(x) cout << #x << " = " << (x) << "\n"
// Macro to print a value and exit successfully.
#define END(e)                                                                 \
  {                                                                            \
    cout << (e) << "\n";                                                       \
    return EXIT_SUCCESS;                                                       \
  }

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  string s1;
  char c;
  cin >> s1;
  /**
   * @block
   * @brief Compares strings character by character in a case-insensitive manner.
   * @details This loop reads the second string one character at a time and compares it
   *          immediately with the corresponding character of the first string.
   * @pre `s1` contains the first string. The input stream contains the second string.
   * @post Prints -1, 1, or 0 and terminates based on the first differing character.
   */
  for (unsigned int i = 0; i < s1.size(); ++i) {
    cin >> c; // Read a single character of the second string.
    if (tolower(s1[i]) < tolower(c))
      END(-1)
    if (tolower(s1[i]) > tolower(c))
      END(1)
  }
  // If the loop finishes, the strings are identical.
  END(0)
}