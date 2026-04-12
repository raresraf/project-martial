/**
 * @file 236A.CPP
 * @brief Discrete classification based on unique character cardinality in a string.
 * 
 * Logic: Determines the 'gender' of a user entity by calculating the number of 
 * distinct characters in their identifier. An even count indicates one class, 
 * while an odd count indicates another.
 * 
 * Algorithm: Frequency-based set membership using a boolean lookup table.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(1)$ due to the fixed-size alphabet/ASCII lookup table.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * Functional Utility: Acts as a bit-vector/lookup table to track character occurrences.
   * Invariant: n[i] is true if ASCII character with code 'i' has been encountered.
   */
  bool n[150];
  memset(n, 0, 150);

  string s;
  getline(cin, s);
  int len = s.size();
  
  // Cardinality of the set of unique characters.
  int c = 0;

  /**
   * Block Logic: Single-pass scan to identify unique symbols.
   * Invariant: 'c' increments only when a character is seen for the first time.
   */
  for (int i = 0; i < len; i++) {
    if (n[s[i]] == 0) {
      n[s[i]] = 1;
      c++;
    }
  }

  /**
   * Functional Utility: Binary classification based on parity of the unique set size.
   */
  if (c % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");

  return 0;
}
