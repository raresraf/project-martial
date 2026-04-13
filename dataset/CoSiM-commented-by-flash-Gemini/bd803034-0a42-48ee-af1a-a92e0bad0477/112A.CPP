/**
 * @file 112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It reads two strings, converts them to lowercase, and then compares them to determine their
 * relative order, outputting -1, 0, or 1.
 * Algorithm: String case conversion using `tolower` followed by `std::string::compare` for lexicographical comparison.
 * Time Complexity: O(L) where L is the maximum length of the input strings, due to character iteration.
 * Space Complexity: O(L) for storing the input strings.
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming, includes many standard libraries.

// Macros for simplified loop constructs.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i) // Loop from 'a' to 'n-1'.
#define REP(i, n) FOR(i, 0, n)                                 // Loop from 0 to 'n-1'.
#define all(a) a.begin(), a.end() // Returns iterators to the beginning and end of a container.
#define pb push_back             // Shorthand for vector's push_back method.

// Type definitions for convenience.
typedef unsigned long long llu; // Alias for unsigned long long.
typedef long long ll;           // Alias for long long.
typedef long double ld;         // Alias for long double.

using namespace std;

int main() {
  string a, b; // Declares two string variables to hold the input strings.
  // Reads the first line (string) from standard input into 'a'.
  getline(cin, a);
  // Reads the second line (string) from standard input into 'b'.
  getline(cin, b);
  /**
   * Block Logic: Iterates through both strings simultaneously to convert all characters to lowercase.
   * This step ensures that the subsequent lexicographical comparison is case-insensitive.
   * Invariant: After this loop, both 'a' and 'b' contain only lowercase characters.
   */
  REP(i, a.size()) {
    a[i] = tolower(a[i]); // Converts character at index 'i' in 'a' to lowercase.
    b[i] = tolower(b[i]); // Converts character at index 'i' in 'b' to lowercase.
  }
  /**
   * Conditional Logic: Compares the two lowercase strings lexicographically using `std::string::compare`.
   * `string::compare` returns 0 if strings are equal, a negative value if 'a' is less than 'b',
   * and a positive value if 'a' is greater than 'b'.
   * The ternary operator then maps these results to -1, 1, or 0 for output.
   */
  if (a.compare(b))
    printf("%s", a.compare(b) < 0 ? "-1\n" : "1\n"); // Prints -1 or 1 based on comparison.
  else
    printf("0\n"); // Prints 0 if strings are equal.
  // Standard program exit.
  return 0;
}
