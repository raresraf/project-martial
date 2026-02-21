/**
 * @file 112A.CPP
 * @brief An idiomatic C++ solution to the "Petya and Strings" problem (Codeforces 112A).
 *
 * @details This program performs a case-insensitive lexicographical comparison of two
 * strings. It leverages the C++ standard library for a clean implementation.
 *
 * @algorithm The approach is to first normalize the strings to a common case and
 * then use standard string comparison operators.
 * 1. A loop iterates through both strings, converting each character to
 *    lowercase in-place using `tolower()`.
 * 2. After normalization, the overloaded `==`, `>`, and `<` operators for
 *    `std::string` are used to perform the lexicographical comparison in a
 *    single step.
 * 3. The result (0, 1, or -1) is stored and then printed.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. This includes O(L)
 *                  for the normalization loop and O(L) for the final comparison.
 * Space Complexity: O(L), to store the two input strings.
 */
#include <iostream>

using namespace std;

int main() {
  string s1 = "";
  string s2 = "";
  int result = 0;
  cin >> s1 >> s2;
  
  /**
   * @brief Block Logic: Normalizes both strings to lowercase.
   * This is the pre-computation step required for a case-insensitive comparison.
   */
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);
  }

  /**
   * @brief Block Logic: Performs lexicographical comparison on the normalized strings.
   * It uses the overloaded operators for `std::string` for a direct comparison.
   */
  if (s1 == s2) {
    result = 0;
  } else if (s1 > s2) {
    result = 1;
  } else if (s1 < s2) {
    result = -1;
  }

  cout << result << endl;

  return 0;
}
