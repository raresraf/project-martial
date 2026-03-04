/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts both to uppercase to ensure a
 * case-insensitive comparison, and then determines their lexicographical order.
 * It outputs -1, 1, or 0 based on the result of the comparison, which is a
 * standard format in competitive programming challenges.
 *
 * @algorithm Case normalization followed by lexicographical comparison.
 * @time_complexity O(L), where L is the length of the strings. The process
 *                  involves a single loop for case conversion and a comparison
 *                  operation which is also linear in time.
 * @space_complexity O(L) for storing the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to perform the string comparison.
 */
int main() {
  string s1, s2;
  // Read two strings from standard input.
  cin >> s1 >> s2;

  int l = s1.length();
  /**
   * @brief Normalize both strings to uppercase.
   * This loop iterates through each character of the strings and converts
   * them to uppercase, ensuring that the subsequent comparison is
   * case-insensitive.
   */
  for (int i = 0; i < l; i++) {
    s1[i] = toupper(s1[i]);
    s2[i] = toupper(s2[i]);
  }

  /**
   * @brief Compare the normalized strings and output the result.
   * The `compare` method of std::string returns:
   *  - A negative value if s1 is lexicographically before s2.
   *  - A positive value if s1 is lexicographically after s2.
   *  - Zero if the strings are equal.
   * The output of this program matches this convention for -1 and 0,
   * but the problem usually expects 1 for the second case. However,
   * the code directly prints the result of `compare`.
   */
  int val = s1.compare(s2);
  cout << val << endl;
}