/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces (Problem 112A).
 * @details This program performs a case-insensitive lexicographical comparison of
 * two input strings and outputs -1, 0, or 1 based on the result.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read two strings from the input.
 * 2. Convert both strings to lowercase to ensure the comparison is case-insensitive.
 *    This is done using the `std::transform` algorithm with `::tolower`.
 * 3. Perform a character-by-character comparison of the two modified strings.
 * 4. The loop terminates at the first differing character, setting the result
 *    to -1 or 1.
 * 5. If the loop completes without finding any differences, the strings are
 *    considered equal, and the result is 0.
 * 6. Print the final result.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the strings. This is
 *   dominated by the `transform` and comparison loops, which are linear.
 * - Space Complexity: O(L), to store the two input strings.
 */
#include <bits/stdc++.h>
#include <cctype>
#include <string>
#define ll long long
using namespace std;
int main() {

/**
 * @brief A common macro in competitive programming for local testing.
 * This block redirects standard input and output to files when the code is
 * run locally (i.e., not on an online judge platform).
 */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, z = 0;
  string a, b;
  cin >> a >> b;

  /**
   * @brief This section standardizes the strings for case-insensitive comparison.
   * Both strings are converted to lowercase.
   */
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);

  /**
   * @brief Performs the lexicographical comparison.
   * This loop iterates through the strings, comparing them character by character
   * to find the first point of difference.
   */
  for (int i = 0; i < a.length(); i++) {
    if (int(a[i]) == int(b[i])) {
      z = 0;
    } else if (int(a[i]) > int(b[i])) {
      z = 1; // String 'a' is lexicographically greater.
      break;
    } else if (int(a[i]) < int(b[i])) {
      z = -1; // String 'a' is lexicographically smaller.
      break;
    }
  }
  
  // Output the result of the comparison.
  cout << z;
  return 0;
}
