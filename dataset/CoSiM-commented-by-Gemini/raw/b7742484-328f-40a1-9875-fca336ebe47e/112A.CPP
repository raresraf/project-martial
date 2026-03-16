/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem (Codeforces 112A).
 *
 * @details
 * This program performs a case-insensitive lexicographical comparison of two strings.
 * The required output is -1 if the first string is smaller, 1 if it's larger,
 * and 0 if they are equal.
 *
 * The approach taken here is to first normalize both input strings by converting them
 * entirely to lowercase. After normalization, the standard `std::string` comparison
 * operators (`>`, `==`, `<`) can be used directly, as they perform the required
 * lexicographical comparison on the modified strings.
 *
 * Algorithm:
 * 1. Read two strings, `a` and `b`.
 * 2. Convert string `a` to lowercase, in-place.
 * 3. Convert string `b` to lowercase, in-place.
 * 4. Use `std::string`'s comparison operators to compare `a` and `b` and print
 *    -1, 0, or 1 accordingly.
 *
 * Time Complexity: O(L), where L is the length of the strings. The normalization
 *                  and comparison steps are all linear in the length of the strings.
 * Space Complexity: O(L), to store the two input strings.
 */
#include <algorithm>
#include <iostream>
#include <string>  // For std::string
#include <cctype>  // For std::tolower
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;

  /**
   * @brief Normalization Step: Convert both strings to lowercase.
   * @details This allows for a simple, direct lexicographical comparison afterward,
   * effectively making the comparison case-insensitive.
   */
  for (int i = 0; i < a.size(); i++)
    a[i] = tolower(a[i]);

  for (int i = 0; i < b.size(); i++)
    b[i] = tolower(b[i]);

  /**
   * @brief Comparison Step: Compare the normalized strings.
   * @details After converting both strings to lowercase, the overloaded C++ string
   * operators provide the exact lexicographical comparison required.
   */
  if (a > b)
    cout << 1;
  else if (a == b)
    cout << 0;
  else
    cout << -1;
  return 0;
}
