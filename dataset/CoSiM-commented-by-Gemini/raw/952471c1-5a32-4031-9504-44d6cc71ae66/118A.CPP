/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and formatting consonants.
 *
 * This program solves the "String Task" problem from Codeforces. It transforms an
 * input string by deleting all vowels, prepending each consonant with a '.', and
 * converting all characters to lowercase. This version uses a large template of
 * includes and constants, common in competitive programming.
 *
 * @algorithm String Transformation / Character Filtering
 * @complexity
 * - Time: O(L), where L is the length of the input string, due to the single
 *   loop that processes each character.
 * - Space: O(L) to store the input string.
 */

// A large set of includes, typical for a competitive programming template.
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

// Constants defined for various potential problems, unused here.
const int inf = (int)2 * 1e9;
const int MOD = (int)1e9 + 7;
const int P = 239;
const int MAX_N = 200000;
const long double pi = 3.1415926;
const long double eps = 1e-6;

int main() {
  // I/O optimizations to speed up cin/cout.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  
  string s;
  cin >> s;

  /**
   * @brief Block Logic: Iterates through the string, applying transformation rules.
   */
  for (int i = 0; i < s.size(); i++) {
    // Convert the character to lowercase in a portable way.
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] += ('a' - 'A');
    
    // Pre-condition: Check if the now-lowercase character is a vowel.
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'e' || s[i] == 'y' ||
        s[i] == 'i' || s[i] == 'u')
      continue; // Skip vowels, effectively deleting them from the output.
    else
      // If it's a consonant, print it prepended with a dot.
      cout << "." << s[i];
  }
  return 0;
}
