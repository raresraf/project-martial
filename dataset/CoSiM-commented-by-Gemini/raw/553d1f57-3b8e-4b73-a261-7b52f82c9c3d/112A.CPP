/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces.
 * @details The problem requires a case-insensitive lexicographical comparison
 * of two strings. The program should output -1, 0, or 1 depending on whether
 * the first string is smaller than, equal to, or greater than the second.
 *
 * @algorithm
 * This solution leverages both C++ string manipulation and C-style string comparison.
 * 1. Read two `std::string` objects.
 * 2. Convert both strings to lowercase using `std::transform` to ensure the
 *    comparison is case-insensitive.
 * 3. Convert the `std::string` objects to C-style character arrays using the
 *    `.c_str()` method.
 * 4. Use the C standard library function `strcmp()` to perform the lexicographical
 *    comparison.
 * 5. `strcmp()` returns a negative value, 0, or a positive value, which directly
 *    corresponds to the problem's required output of -1, 0, or 1 (assuming the
 *    online judge accepts any negative/positive value for -1/1).
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. This is determined
 * by the `transform` and `strcmp` operations, which both take linear time.
 * Space Complexity: O(L) to store the two strings.
 */
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s1, s2;
  cin >> s1 >> s2;
  
  // Convert both strings to lowercase for a case-insensitive comparison.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  
  /**
   * @brief Use the C-style string comparison function `strcmp`.
   * @details `.c_str()` provides a C-style string view of the std::string.
   * `strcmp` returns <0, 0, or >0, which conveniently maps to the
   * problem's required output of -1, 0, or 1.
   */
  cout << strcmp(s1.c_str(), s2.c_str()) << endl;
}
