/**
 * @file 236A.CPP
 * @brief Binary classification based on the cardinality of unique characters in a string.
 * 
 * Logic: Determines the classification of a user ('HER' or 'HIM') by calculating 
 * the number of distinct characters in their identifier. An even count indicates 
 * one class, while an odd count indicates another.
 * 
 * Algorithm: Frequency tracking using an associative map to identify unique symbols.
 * Time Complexity: $O(N \log K)$ where N is the length of the string and K is the 
 * number of unique characters.
 * Space Complexity: $O(K)$ for the associative map storage.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // ans: Cardinality of the set of unique characters.
  int ans = 0;
  string inp;
  // Data Structure: Map to track the presence and frequency of each character.
  map<char, int> character_frequency;
  
  // Functional Utility: Reads the user identifier string.
  if (!(cin >> inp)) return 0;

  /**
   * Block Logic: Single-pass scan to identify unique characters.
   * Invariant: 'ans' increments only when a character is encountered for the first time.
   */
  for (int i = 0; i < (int)inp.length(); i++) {
    character_frequency[inp[i]]++;
    // Logic: First occurrence detection.
    if (character_frequency[inp[i]] == 1)
      ans++;
  }

  /**
   * Functional Utility: Parity-based classification.
   * Logic: Returns the appropriate message based on the unique set size.
   */
  if (ans % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  
  return 0;
}
