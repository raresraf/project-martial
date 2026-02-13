/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem (Codeforces 118A).
 * @author competitive-programmer
 *
 * @details This program processes a string based on the following rules:
 * 1. Deletes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 2. Inserts a '.' character before each consonant.
 * 3. Replaces all uppercase letters with their lowercase equivalents.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through the string using an iterator.
 * 3. For each character, first convert it to lowercase. This simplifies the subsequent
 *    vowel check.
 * 4. Check if the lowercase character is a vowel.
 * 5. If it is NOT a vowel (i.e., it's a consonant), print a '.' character
 *    followed by the consonant.
 *
 * Time Complexity: O(N), where N is the length of the input string, as it is
 * traversed once.
 * Space Complexity: O(N) to store the input string.
 */
#include "bits/stdc++.h"
#define MODULO 1000000007
using namespace std;

// Unused competitive programming typedefs.
typedef unsigned long long ull;
typedef long long ll;

typedef string::iterator sit;

int main() {
  string s;
  cin >> s;
  sit it;

  /**
   * Block Logic: Iterate through each character of the string using an iterator.
   */
  for (it = s.begin(); it != s.end(); it++) {
    /**
     * Logic: Convert any uppercase character to its lowercase equivalent.
     * `*it < 97` checks if the ASCII value is less than 'a', which is true
     * for uppercase English letters.
     */
    if (*it < 97)
      *it += 32;
    
    /**
     * Pre-condition: Check if the character is NOT a vowel.
     * The vowels are defined as 'a', 'e', 'y', 'u', 'o', 'i'.
     */
    if (!(*it == 'a' || *it == 'e' || *it == 'y' || *it == 'u' || *it == 'o' ||
          *it == 'i'))
      // If it's a consonant, print a dot followed by the character.
      printf("%c%c", '.', *it);
  }

  return 0;
}
