/**
 * @file 112A.CPP
 * @brief Idiomatic C++ solution for the "Petya and Strings" problem (Codeforces 112A).
 * @details This program compares two strings lexicographically, ignoring the case of the letters,
 * and prints -1, 1, or 0 based on the comparison result.
 *
 * Algorithm: Normalization and Direct Comparison
 * The program reads two strings. It first iterates through both strings to convert all
 * characters to lowercase, ensuring a case-insensitive comparison. After this
 * normalization step, it directly uses the overloaded C++ std::string comparison
 * operators (`>`, `<`, `==`) to perform the lexicographical comparison and print the result.
 * This is a clean and idiomatic approach.
 *
 * Time Complexity: O(L), where L is the length of the strings. This is dominated by the
 * normalization loop and the single comparison, which in the worst case checks all characters.
 * Space Complexity: O(L) to store the two input strings.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  // c: the first input string.
  // s: the second input string.
  string c, s;
  cin >> c;
  cin >> s;
  
  // n: an unused integer variable.
  int n = 0;
  
  /**
   * Block Logic: Normalize both strings to lowercase for case-insensitive comparison.
   * Pre-condition: 'c' and 's' hold the input strings.
   * Invariant: After the loop, all characters in 'c' and 's' are lowercase.
   */
  for (int i = 0; i < c.length(); i++) {
    c[i] = tolower(c[i]);
    s[i] = tolower(s[i]);
  }
  
  /**
   * Core Logic: Use std::string's overloaded operators for lexicographical comparison.
   * This is the idiomatic C++ way to compare strings after normalization.
   */
  if (c > s) {
    cout << "1";
  } else if (c < s) {
    cout << "-1";
  } else if (c == s) {
    cout << "0";
  }

  return 0;
}