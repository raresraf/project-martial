/**
 * @file 281A.CPP
 * @brief Solution for a string capitalization problem, likely Codeforces 281A "Word Capitalization".
 * @details This program reads a single word and capitalizes its first letter. If the
 * first letter is already capitalized, it remains unchanged.
 *
 * @section
 *
 * Algorithm:
 * The program reads a string from standard input. It then accesses the first character
 * of the string at index 0 and applies the `toupper()` function from the cctype library.
 * This function converts a lowercase letter to uppercase, leaving other characters
 * (like existing uppercase letters or numbers) unchanged. The modified string is then
 * printed to standard output.
 *
 * - Time Complexity: O(L) where L is the length of the string, dominated by I/O operations.
 *   The capitalization itself is an O(1) operation.
 * - Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  /**
   * Block Logic: Reads a word from standard input. The loop structure would
   * process multiple words if provided, but the problem typically specifies one.
   */
  while (cin >> s) {
    // Inline Logic: Capitalizes the first character of the string in-place.
    s[0] = toupper(s[0]);
    cout << s << endl;
  }
  return 0;
}
