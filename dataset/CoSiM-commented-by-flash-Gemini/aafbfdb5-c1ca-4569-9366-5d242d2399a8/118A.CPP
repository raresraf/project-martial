/**
 * @file 118A.CPP
 * @brief Logic for string normalization and vocalic character deletion.
 * 
 * Logic: Implements a specific string transformation pipeline:
 * 1. Normalizes all uppercase characters to lowercase.
 * 2. Filters out all vocalic characters (a, e, i, o, u, y).
 * 3. Prefixes each remaining character (consonant) with a '.' separator.
 * 
 * Algorithm: Two-pass sequential transformation (normalization then filtering).
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  // Functional Utility: Reads the input character sequence.
  cin >> s;

  /**
   * Block Logic: Initial pass for case normalization.
   * Logic: Maps uppercase ASCII [65-90] to lowercase [97-122] by adding 32.
   */
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] >= 65 && s[i] <= 90) {
      s[i] = s[i] + 32;
    }
  }

  /**
   * Block Logic: Secondary pass for vocalic filtering and formatting.
   * Logic: Evaluates each normalized character against the vocalic set {a, e, i, o, u, y}.
   * Consonants are preserved and output with a leading '.' delimiter.
   */
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y') {
      cout << "." << s[i];
    }
  }

  // Functional Utility: Terminates the output stream with a newline.
  cout << endl;

  return 0;
}
