/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - Way Too Long Words.
 *
 * This program takes a list of words and abbreviates any word that is
 * strictly longer than 10 characters. The abbreviation consists of the
 * first letter of the word, followed by the number of characters between
 * the first and last letters, and then the last letter of the word.
 * Words 10 characters or shorter are printed as is.
 *
 * Algorithm: String length check and conditional formatting. For each word,
 * its length is checked. If it exceeds 10, it's abbreviated; otherwise, it's printed whole.
 * Time Complexity: O(N * L), where N is the number of words and L is the average
 *                  length of a word (as string operations like strlen() are O(L)).
 * Space Complexity: O(L_max) for storing the longest word.
 */

#include <algorithm> // Not used in this specific solution's main logic.
#include <cmath>     // Not used.
#include <cstdio>    // Required for scanf, printf.
#include <cstring>   // Required for strlen.
#include <iostream>  // Not used (uses cstdio).
#include <map>       // Not used.
#include <queue>     // Not used.
#include <stack>     // Not used.
#include <string>    // Not used (uses char arrays).
#include <utility>   // Not used.
#include <vector>    // Not used.

// Common competitive programming macros and aliases for brevity.
#define fi first          // Alias for pair.first (unused here).
#define se second         // Alias for pair.second (unused here).
#define pb push_back      // Alias for vector.push_back (unused here).
#define mp make_pair      // Alias for std::make_pair (unused here).
#define pi 2 * acos(0.0)  // Value of PI (unused here).
#define eps 1e-9          // Epsilon for floating point comparisons (unused here).
#define PII pair<int, int> // Alias for pair<int, int> (unused here).
#define PDD pair<double, double> // Alias for pair<double, double> (unused here).
#define LL long long      // Alias for long long integer (unused here).

using namespace std; // Uses the standard namespace.

// Declare 'N' for the number of words (global).
// Declare 'len' for the length of the current word (global).
// Declare 'msk' as a character array to store the current word (max 200 chars).
int N, len;
char msk[200];

int main() {
  /**
   * @brief Main function to execute the Way Too Long Words problem logic.
   *
   * Reads an integer 'N' representing the number of words to process.
   * Then, it enters a loop to read 'N' words one by one. For each word,
   * it calculates its length and applies the abbreviation logic: if the word
   * is longer than 10 characters, it's abbreviated; otherwise, it's printed as is.
   */
  // Read the total number of words to be processed.
  scanf("%d", &N);
  // Block Logic: Process each word.
  // Invariant: 'N' decreases with each iteration, processing one word per loop.
  while (N--) {
    // Read the current word into the 'msk' character array.
    scanf("%s", msk);
    // Calculate the length of the current word.
    len = strlen(msk);
    // Conditional Logic: Check if the word needs abbreviation.
    // Pre-condition: 'len' holds the length of the string 'msk'.
    if (len < 11)
      // If the word is 10 characters or shorter, print it as is, followed by a newline.
      printf("%s\n", msk);
    else
      // If the word is longer than 10 characters, print the abbreviated word:
      // first character, count of middle characters (len - 2), last character.
      printf("%c%d%c\n", msk[0], len - 2, msk[len - 1]);
  }
  return 0; // Indicate successful program execution.
}
