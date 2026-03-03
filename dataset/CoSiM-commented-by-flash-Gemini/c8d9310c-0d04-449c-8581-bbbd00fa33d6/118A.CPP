/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A - "String Task".
 * This program transforms an input word by converting all uppercase letters to lowercase,
 * removing all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and then inserting a '.' before
 * each remaining consonant.
 * Algorithm: Character-by-character processing with conditional transformation and appending.
 * Time Complexity: O(L), where L is the length of the input word. The string is iterated once.
 * Space Complexity: O(L) - to store the input and result strings. The result string can be up to
 *                  twice the length of the input string in the worst case.
 */
#define _CRT_SECURE_NO_WARNINGS // Suppresses security warnings for functions like scanf, strcpy (common in competitive programming).
#include <algorithm> // Provides functions like std::tolower, although not directly used here.
#include <deque>     // Double-ended queue container (not used in this specific solution).
#include <functional> // Function objects (not used in this specific solution).
#include <iostream>  // Standard input/output stream library (used indirectly via stdio.h).
#include <map>       // Map container (not used in this specific solution).
#include <queue>     // Queue container (not used in this specific solution).
#include <set>       // Set container (not used in this specific solution).
#include <stack>     // Stack container (not used in this specific solution).
#include <stdio.h>   // C-style input/output functions (scanf, printf).
#include <string.h>  // C-style string manipulation functions (strlen).
#include <string>    // C++ string class (std::string).
#include <tuple>     // Tuple (not used in this specific solution).
#include <vector>    // Vector container (not used in this specific solution).
using namespace std; // Uses the standard namespace.

// Macros for common competitive programming constants and types.
#define INF 987654321      // A large integer value representing infinity.
#define mod 1000000007     // A common modulo value for arithmetic operations.
#define mt make_tuple      // Shorthand for std::make_tuple.
#define mp make_pair       // Shorthand for std::make_pair.
typedef long long ll;      // Alias for long long.

int main() {
  char a[101];  // Character array to store the input word. Max length 100 + null terminator.
  char b[201];  // Character array to store the transformed word. Max length 100 * 2 + null terminator.
  scanf("%s", a); // Reads the input word from standard input (C-style string).

  int len = strlen(a); // Calculates the length of the input word.
  int top = 0;         // Index for the 'b' array, tracking the current length of the transformed string.

  /**
   * Block Logic: Iterates through each character of the input word, applying transformations.
   * - Converts uppercase letters to lowercase.
   * - Skips vowels.
   * - Prepends a '.' to each consonant and appends the consonant to the result string 'b'.
   * Invariant: `b` correctly stores the transformed string built up to the current character `a[i]`.
   */
  for (int i = 0; i < len; i++) { // Loop through each character of the input word.
    // Converts uppercase letters to lowercase.
    if (a[i] >= 'A' && a[i] <= 'Z')
      a[i] = a[i] + 'a' - 'A'; // ASCII conversion: 'A'-'Z' are 65-90, 'a'-'z' are 97-122. Difference is 32.

    // Checks if the character is a vowel (a, e, i, o, u, y).
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'u' ||
        a[i] == 'o' || a[i] == 'y') {
      // If it's a vowel, do nothing (effectively removing it).
    } else {
      // If it's a consonant:
      b[top++] = '.'; // Prepend a '.' to the result string.
      b[top++] = a[i]; // Append the consonant to the result string.
    }
  }
  b[top] = '\0'; // Null-terminate the transformed string 'b'.
  printf("%s\n", b); // Prints the transformed string to standard output, followed by a newline.

  return 0; // Indicates successful program execution.
}