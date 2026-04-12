/**
 * @file 118A.CPP
 * @brief Logic for string transformation via vocalic filtering and case normalization.
 * 
 * Logic: Implements a specific string Task:
 * 1. Normalizes all characters to lowercase.
 * 2. Filters out all vocalic characters (a, e, i, o, u, y).
 * 3. Prefixes every remaining consonant with a dot (.) separator.
 * 
 * Algorithm: Single-pass scan with set-based character exclusion and case-normalization.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for storing the transformed result string.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  string s;
  // Functional Utility: Reads the input character sequence.
  cin >> s;
  
  string ans = "";
  /**
   * Data Structure: Acts as a lookup table for vocalic characters to be excluded.
   */
  vector<char> v;
  v.push_back('a');
  v.push_back('e');
  v.push_back('i');
  v.push_back('o');
  v.push_back('u');
  v.push_back('y');

  /**
   * Block Logic: Sequential transformation of the input string.
   * Invariant: Each character is processed for case normalization and vocalic classification.
   */
  for (int i = 0; i < s.length(); i++) {
    char temp;
    /**
     * Logic: Normalizes uppercase ASCII characters [A-Z] to lowercase [a-z] 
     * by adding the 32 offset.
     */
    if (s[i] >= 'A' && s[i] <= 'Z')
      temp = s[i] + 32;
    else
      temp = s[i];

    /**
     * Functional Utility: Filters the character based on the vocalic set.
     * Logic: If the normalized character is not in the set {a, e, i, o, u, y}, 
     * it is treated as a consonant and formatted into the output string.
     */
    if (find(v.begin(), v.end(), temp) == v.end()) {
      ans += ".";
      ans += temp;
    }
  }
  
  // Output the final resulting sequence.
  cout << ans;
  
  return (0);
}
