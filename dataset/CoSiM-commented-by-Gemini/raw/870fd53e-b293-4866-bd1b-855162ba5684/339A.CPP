/*
Codeforces Round #197 (Div. 2) - A. Helpful Maths
https://codeforces.com/problemset/problem/339/A
*/
/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (339A).
 *
 * @details
 * The program takes a string representing a sum (e.g., "3+2+1") and reorders
 * the numbers in non-decreasing order (e.g., "1+2+3").
 *
 * This particular implementation uses a confusing but functional method involving
 * arithmetic on ASCII character codes to achieve the result.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Create a vector of integers.
 * 3. Iterate through the string, and for each character that is a digit:
 *    a. Perform the operation `s[i] + '0'`, which adds the ASCII value of '0'
 *       to the ASCII value of the digit character, and push the resulting
 *       integer into the vector. (e.g., for '1', this is 49 + 48 = 97).
 * 4. Sort the vector of integers. Since the transformation preserves order, this
 *    effectively sorts the original digits.
 * 5. Reconstruct the output string:
 *    a. For each integer in the sorted vector, subtract the ASCII value of '0'
 *       to get back the original digit's ASCII value (e.g., 97 - 48 = 49).
 *    b. Convert this integer back to a character and append it to the result string.
 *    c. Append a '+' sign, except for the last number.
 * 6. Print the final sorted string.
 *
 * Time Complexity: O(L log L), where L is the length of the string, dominated by sorting.
 * Space Complexity: O(L) for the vector and the output string.
 */
#include <bits/stdc++.h>
using namespace std;
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

int main() {
  FAST_INP;
  string s;
  cin >> s;
  vector<int> v;
  /**
   * @brief Extracts digits from the input string using non-standard arithmetic.
   * For each digit character, its ASCII value is added to the ASCII value of '0'
   * and the result is stored. This preserves the relative order of the digits.
   */
  for (int i = 0; i < s.size(); i++) {
    if (isdigit(s[i]))
      v.push_back(s[i] + '0');
  }
  // Sort the vector of transformed numbers.
  sort(v.begin(), v.end());
  string ans = "";
  /**
   * @brief Reconstructs the string from the sorted, transformed numbers.
   */
  for (int i = 0; i < v.size(); i++) {
    // Reverses the arithmetic from the first loop to get the digit character back,
    // then appends it to the answer string.
    ans += v[i] - '0';
    if (i != v.size() - 1)
      ans += "+";
  }
  cout << ans << "\n";
  return 0;
}
