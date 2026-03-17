/**
 * @69a91490-f8b3-4094-843c-c533b4d2d007/112A.CPP
 * @brief Problem: Petya and Strings.
 *        Compares two strings lexicographically, ignoring case.
 *        Outputs -1 if the first string is lexicographically smaller,
 *        1 if the second string is lexicographically smaller,
 *        and 0 if the strings are equal.
 * Algorithm: Character-by-character comparison with case normalization.
 *            Iterates through both strings simultaneously, converting characters to lowercase.
 *            Compares characters at each position and determines lexicographical order.
 *            If characters are different, sets a result flag and breaks.
 * Time Complexity: O(L) where L is the length of the shorter string, as comparison stops at the first difference.
 * Space Complexity: O(L_max) where L_max is the maximum length of the input strings,
 *                   for storing the strings themselves.
 */
#include <ctype.h>     // Functional Utility: Provides tolower() for case conversion.
#include <iostream>    // Functional Utility: Provides cin and cout for input/output.
#include <string>      // Functional Utility: Provides string class for string manipulation.
using namespace std;

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string a, b;
  int i, r = 0; // Invariant: 'r' stores the comparison result: 0 for equal, 1 for 'a' greater, -1 for 'b' greater.
  // Block Logic: Reads the first string from standard input.
  cin >> a;
  // Block Logic: Reads the second string from standard input.
  cin >> b;
  
  // Block Logic: Iterates through the strings character by character until a null terminator is reached in string 'a'
  // or a lexicographical difference is found.
  // Pre-condition: Both strings 'a' and 'b' are populated.
  // Invariant: 'a[i]' and 'b[i]' are converted to lowercase for case-insensitive comparison.
  // Post-condition: 'r' holds the comparison result, or remains 0 if strings are equal up to the length of 'a'.
  for (i = 0; a[i] != '\0'; i++) {
    // Functional Utility: Converts characters to lowercase for case-insensitive comparison.
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
    // Block Logic: Compares the lowercase characters at the current position.
    // Pre-condition: 'a[i]' and 'b[i]' are normalized to lowercase.
    if (a[i] > b[i]) {
      r = 1; // Functional Utility: Sets result to 1 if 'a' is lexicographically greater.
      break; // Functional Utility: Terminates loop early as order is determined.
    } else if (b[i] > a[i]) {
      r = -1; // Functional Utility: Sets result to -1 if 'b' is lexicographically greater.
      break; // Functional Utility: Terminates loop early as order is determined.
    }
  }
  // Functional Utility: Prints the lexicographical comparison result.
  cout << r;
  return 0; // Functional Utility: Indicates successful program execution.
}
