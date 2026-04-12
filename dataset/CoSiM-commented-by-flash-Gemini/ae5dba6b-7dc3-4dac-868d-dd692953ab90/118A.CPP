/**
 * @file 118A.CPP
 * @brief Logic for string transformation involving vocalic filtering and consonant normalization.
 * 
 * Logic: Implements a specific string processing algorithm:
 * 1. Iterates through each character of the input.
 * 2. Normalizes the character to lowercase.
 * 3. Discards the character if it belongs to the vocalic set {a, e, i, o, u, y}.
 * 4. Prefixes preserved characters (consonants) with a '.' separator.
 * 
 * Algorithm: Single-pass scan with linear state transformation.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for the input string buffer.
 */

#include <bits/stdc++.h>
#include <iostream>
#include <math.h>

using namespace std;

int main() {
  string str;
  // Functional Utility: Reads the raw character sequence.
  cin >> str;

  /**
   * Block Logic: Main transformation loop.
   * Invariant: Evaluates each character independently for classification and normalization.
   */
  for (int i = 0; i < str.length(); i++) {
    // Normalization: Maps uppercase ASCII to lowercase.
    char c = (char)tolower(str[i]);
    
    /**
     * Functional Utility: Conditional filtering and formatting.
     * Logic: If 'c' is not a vowel, it is formatted as '.c' in the output stream.
     */
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')
      cout << "." << c;
  }
  
  return 0;
}
