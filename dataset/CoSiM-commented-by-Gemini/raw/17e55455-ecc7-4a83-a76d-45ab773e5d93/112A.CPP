/**
 * @file 112A.CPP
 * @brief Solution for a problem requiring case-insensitive lexicographical string comparison.
 * @details This program reads two strings and compares them lexicographically, ignoring differences in case.
 * It outputs -1 if the first string is lexicographically smaller, 1 if it is larger, and 0 if they are equal.
 *
 * @section algorithm Algorithm
 * 1. Read two strings, `s` and `p`.
 * 2. Convert both strings to the same case (lowercase in this implementation) to ensure the comparison is case-insensitive.
 *    - This is done by iterating through each character of both strings.
 *    - If a character is uppercase, it is converted to lowercase by adding 32 to its ASCII value.
 * 3. After conversion, the two strings are compared using standard lexicographical comparison operators.
 * 4. The program then prints the result based on the comparison:
 *    - `0` if the strings are identical.
 *    - `-1` if the first string comes before the second in lexicographical order.
 *    - `1` if the first string comes after the second.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the strings. This is because the program iterates through the strings once for conversion and once for comparison.
 * - Space Complexity: O(L), to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

#define fast ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0)

/**
 * @brief Main function to perform the case-insensitive string comparison.
 * @return 0 on successful execution.
 */
int main() {
  fast;
  string s, p;
  cin >> s >> p;

  // Convert both strings to lowercase for case-insensitive comparison.
  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
    p[i] = tolower(p[i]);
  }

  // Perform lexicographical comparison on the lowercase strings.
  if (s == p) {
    cout << 0 << endl;
  } else if (s < p) {
    cout << -1 << endl;
  } else {
    cout << 1 << endl;
  }

  return 0;
}
