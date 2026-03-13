/**
 * @file 112A.CPP
 * @brief Solution to "Petya and Strings" problem from Codeforces (https://codeforces.com/problemset/problem/112/A)
 * 
 * @details
 * This program performs a case-insensitive lexicographical comparison of two input strings.
 * It converts both strings to lowercase and then compares them.
 *
 * @algorithm Case-insensitive string comparison
 * @time_complexity O(L), where L is the length of the strings. This is dominated by the `transform` and comparison operations.
 * @space_complexity O(L) for storing the two strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int a; // Stores the result of the comparison: -1, 0, or 1.
  string i1, i2; // Input strings.
  cin >> i1 >> i2;

  /**
   * @brief Block to normalize strings to lowercase.
   * `transform` is used to apply the `::tolower` function to every character
   * in both input strings, ensuring the subsequent comparison is case-insensitive.
   */
  transform(i1.begin(), i1.end(), i1.begin(), ::tolower);
  transform(i2.begin(), i2.end(), i2.begin(), ::tolower);

  /**
   * @brief Perform lexicographical comparison.
   * The ternary operator directly maps the result of the string comparison
   * to the required output values (-1, 0, 1).
   * Pre-condition: i1 and i2 are now in lowercase.
   * Post-condition: `a` holds the lexicographical comparison result.
   */
  a = i1 > i2 ? 1 : (i1 < i2 ? -1 : 0);
  
  // Output the final comparison result.
  cout << a;
  return 0;
}
