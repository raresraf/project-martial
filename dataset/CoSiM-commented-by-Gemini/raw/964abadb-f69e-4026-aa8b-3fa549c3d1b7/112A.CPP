/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program solves the "Petya and Strings" problem from Codeforces. It reads
 * two strings, converts them to lowercase using the `std::transform` algorithm,
 * and then compares them using standard string operators to determine their
 * lexicographical order.
 *
 * @algorithm STL Transform for Case Conversion
 * @complexity
 * - Time: O(L), where L is the length of the strings. The `transform` algorithm
 *   and the final comparison both take linear time.
 * - Space: O(L) to store the two input strings. The conversion is done in-place.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // I/O optimizations for speed.
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s, s1;
  cin >> s >> s1;

  /**
   * @brief Functional Utility: Converts strings to lowercase using `std::transform`.
   * This applies the `::tolower` function to every character in the string,
   * performing an efficient, in-place case conversion.
   */
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  
  /**
   * @brief Block Logic: Compares the lowercase strings and prints the result.
   * Overloaded operators for `std::string` provide a clean way to perform
   * lexicographical comparison.
   */
  if (s > s1)
    cout << "1\n";
  if (s == s1)
    cout << "0\n";
  if (s < s1)
    cout << "-1\n";
}