/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details This program processes a string by converting it to lowercase,
 * removing all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and inserting a '.'
 * before each remaining consonant. This implementation builds the result in a
 * separate character vector before printing.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Create an empty character vector to store the output.
 * 3. Iterate through each character of the input string.
 * 4. In the loop, convert the character to lowercase.
 * 5. Check if the character is a consonant.
 * 6. If it is, push a '.' character followed by the consonant itself onto the
 *    output vector.
 * 7. After the loop, print all characters in the output vector sequentially.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string, as it
 * involves a single pass to build the vector and another to print it.
 * Space Complexity: O(N) to store the input string and approximately O(2N) for
 * the output vector in the worst case (if all characters are consonants).
 */

#include "bits/stdc++.h"
// A collection of personal comments and prayers, common in competitive programming.
// PRAY :DOLPHINGARLIC: orz orz orz orz orz
// PRAY :DORI: orz orz orz orz orz orz orz
// PRAY :SAHIL KUCHLOUS: orz orz orz orz orz
// pray :stefan: orz orz orz orz orz orz orz
// pray :foshy: orz orz orz orz orz orz orz

// A collection of personal macros for competitive programming shortcuts.
#define GS                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(nullptr);                                                            \
  cout.tie(0)
#define all(x) x.begin(), x.end()
#define el '\n'
#define Int long long
#define elif else if
#define test                                                                   \
  Int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)
// #define

using namespace std;

int main() {
  // GS is a macro for fast I/O setup.
  GS;
  string s;
  cin >> s;
  vector<char> v; // A vector to build the output string.

  /**
   * @block
   * @brief Process the input string and build the output vector.
   */
  for (int i{}; i < (int)s.size(); i++) {
    // Convert the character to lowercase.
    s[i] = tolower(s[i]);
    // Check if the character is a vowel.
    if (s[i] != 'a' && s[i] != 'o' && s[i] != 'y' && s[i] != 'e' &&
        s[i] != 'u' && s[i] != 'i') {
      // If it's a consonant, add a '.' and the character to the result vector.
      v.push_back('.');
      v.push_back(s[i]);
    }
  }

  /**
   * @block
   * @brief Print the contents of the result vector.
   */
  for (auto i : v) {
    cout << i;
  }
  cout << endl;
  return 0;
}
/*
Timestamp from original author.
12:43 AM
9/18/2020
*/