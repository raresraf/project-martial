/**
 * @file 118A.CPP
 * @brief Implements a string transformation algorithm to convert a given word into a "Pangram" format.
 * * Algorithm: Character filtering and transformation.
 * Time Complexity: $O(N)$ where N is the length of the input string, due to single-pass iteration.
 * Space Complexity: $O(1)$ excluding input/output storage.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

// Standard competitive programming constants, not directly used in this specific algorithm's core logic.
const int inf = (int)2 * 1e9;
const int MOD = (int)1e9 + 7;
const int P = 239;
const int MAX_N = 200000;
const long double pi = 3.1415926;
const long double eps = 1e-6;

int main() {
  // Functional Utility: These lines are typically used in competitive programming to speed up I/O operations.
  // `ios_base::sync_with_stdio(0)` unties C++ standard streams from C standard streams.
  // `cin.tie(0)` unties cin from cout, preventing flushes of cout before each cin operation.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  string s;
  cin >> s;
  /**
   * Block Logic: Iterates through the input string, processing each character.
   * Pre-condition: Input string `s` is populated.
   * Invariant: Characters are processed one by one; vowels are skipped, consonants are transformed and printed.
   * Post-condition: A modified string consisting of only consonants (prefixed with '.') is printed to standard output.
   */
  for (int i = 0; i < s.size(); i++) {
    // Inline: Converts uppercase letters to lowercase.
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] += ('a' - 'A');
    // Block Logic: Checks if the character is a vowel (a, o, e, y, i, u).
    // Invariant: If a vowel, the character is skipped; otherwise, it is treated as a consonant.
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'e' || s[i] == 'y' ||
        s[i] == 'i' || s[i] == 'u')
      continue;
    else
      // Inline: Prints a '.' prefix followed by the consonant.
      cout << "." << s[i];
  }
  return 0;
}