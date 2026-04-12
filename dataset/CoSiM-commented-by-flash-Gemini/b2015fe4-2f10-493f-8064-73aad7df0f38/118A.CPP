/**
 * @file 118A.CPP
 * @brief Logic for string transformation via vocalic filtering and case normalization.
 * 
 * Logic: Implements a specific string processing algorithm:
 * 1. Normalizes characters to lowercase.
 * 2. Filters out all vocalic characters {a, o, y, e, u, i}.
 * 3. Prefixes each remaining character (consonant) with a dot (.) separator.
 * 
 * Algorithm: Single-pass scan with ASCII-based state normalization.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for the input string buffer.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string s;
  // Functional Utility: Reads the source character sequence.
  if (!(cin >> s)) return 0;

  /**
   * Block Logic: Main transformation loop.
   * Invariant: Evaluates each character independently for classification and normalization.
   */
  for (int i = 0; i < (int)s.length(); i++) {
    /**
     * Logic: Case normalization.
     * Maps uppercase ASCII (<= 90) to lowercase by adding 32.
     */
    if (s[i] <= 90)
      s[i] += 32;

    /**
     * Functional Utility: Conditional filtering.
     * Logic: If character belongs to the vocalic set {a, o, y, e, u, i}, it is omitted.
     * Otherwise, it is formatted as '.c' in the output stream.
     */
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      continue;
    } else {
      cout << "." << s[i];
    }
  }
  
  return 0;
}
