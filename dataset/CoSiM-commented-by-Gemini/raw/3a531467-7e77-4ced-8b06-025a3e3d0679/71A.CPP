/**
 * @file 71A.CPP
 * @brief Abbreviates words with a length greater than 10.
 *
 * This program reads a series of words and applies a shortening rule to them.
 * If a word is longer than 10 characters, it's replaced by an abbreviation
 * formed from its first character, the count of its internal characters, and
 * its last character. Otherwise, the word is printed as is.
 * This implementation uses C-style I/O (`scanf`, `printf`) and strings.
 *
 * Algorithm:
 * 1. Read an integer `N`, the total number of words to process.
 * 2. Loop `N` times:
 *    a. Read a word into a character array `msk`.
 *    b. Calculate the length of the word using `strlen`.
 *    c. If the length is less than 11, print the word unchanged.
 *    d. If the length is 11 or more, print the abbreviation in the format:
 *       first character, length minus two, last character.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum
 * length of a word, dominated by the I/O operations.
 * Space Complexity: O(L_max), where L_max is the fixed size of the character
 * array buffer (200).
 *
 * @note This file includes a large number of headers and macro definitions
 *       that are common in competitive programming templates but are not used
 *       in this specific problem.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// Unused competitive programming macros.
#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long

using namespace std;

int N, len;
char msk[200]; // Buffer to hold the input word.

int main() {
  // Read the number of words to process.
  scanf("%d", &N);
  
  /**
   * @brief Main loop to process each word.
   * Pre-condition: `N` is the number of words to read.
   */
  while (N--) {
    scanf("%s", msk);
    len = strlen(msk);
    
    // Check if the word is "too long".
    if (len < 11)
      printf("%s\n", msk);
    else
      // Print the abbreviation for long words.
      printf("%c%d%c\n", msk[0], len - 2, msk[len - 1]);
  }
  return 0;
}
