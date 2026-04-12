/**
 * @file 118A.CPP
 * @brief Logic for string transformation involving vocalic deletion and character normalization.
 * 
 * Logic: Implements a multi-stage string processing task:
 * 1. Filters out all vocalic characters {A, E, I, O, U, Y} (case-insensitive).
 * 2. Normalizes remaining characters (consonants) to lowercase.
 * 3. Prefixes every preserved character with a dot (.) separator.
 * 
 * Algorithm: Multi-pass sequential transformation.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(L)$ for the fixed-size intermediate character buffer (size 101).
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // Static buffers for source and intermediate filtered strings.
  char s[101];
  char temp[101];
  int j = 0;
  
  // Functional Utility: Reads the input string from standard input.
  if (!(cin >> s)) return 0;

  /**
   * Block Logic: Initial filtering pass (Vocalic Deletion).
   * Invariant: Evaluates each character against the vocalic set and 
   * populates 'temp' with non-vocalic characters.
   */
  for (int i = 0; s[i] != '\0'; i++) {
    if (s[i] != 'A' && s[i] != 'a' && s[i] != 'E' && s[i] != 'e' &&
        s[i] != 'I' && s[i] != 'i' && s[i] != 'O' && s[i] != 'o' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'y' && s[i] != 'Y') {
      temp[j] = s[i];
      j++;
    }
  }
  // Termination: Correctly null-terminates the filtered intermediate string.
  temp[j] = '\0';

  /**
   * Block Logic: Case normalization pass.
   * Logic: Maps uppercase ASCII characters [A-Z] to lowercase by adding 32.
   */
  for (int i = 0; temp[i] != '\0'; i++) {
    if (temp[i] >= 'A' && temp[i] <= 'Z') {
      temp[i] = temp[i] + 32;
    }
  }

  /**
   * Block Logic: Final formatting and output pass.
   * Logic: Interleaves the dot separator with each normalized consonant.
   */
  for (int i = 0; temp[i] != '\0'; i++) {
    cout << "." << temp[i];
  }
  
  return 0;
}
