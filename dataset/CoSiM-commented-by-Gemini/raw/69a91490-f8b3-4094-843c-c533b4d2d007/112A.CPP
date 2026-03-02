#include <ctype.h>
#include <iostream>
using namespace std;

/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem from Codeforces (Problem 112A).
 * @details This program performs a case-insensitive lexicographical comparison of two
 * strings by iterating through them character by character.
 *
 * Algorithm:
 * The program reads two strings. It then iterates through them simultaneously, character
 * by character. In each step, it converts the corresponding characters from both strings
 * to lowercase to ensure a case-insensitive comparison. It then compares these two
 * characters. If they differ, the lexicographical relationship is determined (`-1` if the
 * first string is smaller, `1` if it's larger), the result is stored, and the loop is
 * terminated. If the characters are the same, the loop continues. If the loop completes
 * without finding any differences, the strings are considered equal, and the result is 0.
 *
 * Time Complexity: O(L), where L is the length of the strings. The loop runs up to the
 * length of the strings.
 * Space Complexity: O(L) to store the two input strings.
 */
int main() {
  string a, b;
  // `r` stores the result: -1, 0, or 1.
  int i, r = 0;
  cin >> a;
  cin >> b;

  /**
   * @brief Loop through the strings, comparing character by character.
   * The loop condition `a[i] != '\0'` is a C-style check for the end of the string.
   */
  for (i = 0; a[i] != '\0'; i++) {
    // Convert characters to lowercase for case-insensitive comparison.
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
    
    // If a difference is found, set the result and break the loop.
    if (a[i] > b[i]) {
      r = 1;
      break;
    } else if (b[i] > a[i]) {
      r = -1;
      break;
    }
  }
  // Print the final result.
  cout << r;
  return 0;
}
