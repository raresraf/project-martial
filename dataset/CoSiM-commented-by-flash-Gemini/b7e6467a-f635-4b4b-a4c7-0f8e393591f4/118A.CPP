/**
 * @file 118A.CPP
 * @brief Logic for string transformation involving vocalic filtering and consonant formatting.
 * 
 * Logic: Implements a specific string processing algorithm:
 * 1. Scans each character of the input.
 * 2. Filters out all vocalic characters based on a predefined set {A, O, Y, E, U, I} (case-insensitive).
 * 3. Normalizes remaining characters (consonants) to lowercase and prefixes them with a '.'.
 * 
 * Algorithm: Single-pass scan with nested-loop character classification.
 * Time Complexity: $O(N)$ where N is the length of the string (set size is constant).
 * Space Complexity: $O(L)$ for the fixed-size character buffer (size 100).
 */

#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  // Static buffer for input string.
  char s[100];
  // Static lookup set for vocalic characters to be filtered.
  char a[13] = "AOYEUIaoyeui";
  int found_vowel = 0;
  char current_char;

  // Functional Utility: Reads the input string from standard input.
  if (fscanf(stdin, "%s", s) != 1) return 0;

  /**
   * Block Logic: Iterative transformation of the input character sequence.
   * Invariant: Evaluates each character independently for vocalic membership.
   */
  for (int i = 0; s[i] != '\0'; ++i) {
    /**
     * Block Logic: Character classification.
     * Logic: Performs a linear search in the vowel set 'a' to identify matches.
     */
    for (int j = 0; a[j] != '\0'; ++j) {
      if (s[i] == a[j]) {
        found_vowel = 1;
        break;
      }
    }

    /**
     * Functional Utility: Conditional formatting for non-vocalic characters.
     * Logic: If the character was not found in the vowel set, it is normalized 
     * to lowercase and output with a preceding '.' separator.
     */
    if (!found_vowel) {
      current_char = s[i];
      fprintf(stdout, ".%c", (char)tolower(current_char));
    }
    
    // State Reset: Clears the flag for the next character iteration.
    found_vowel = 0;
  }
  
  return 0;
}
