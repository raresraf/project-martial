/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces.
 * @details The problem requires comparing two strings lexicographically,
 * ignoring differences in case. The program should output -1 if the first
 * string is less than the second, 1 if it's greater, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings from standard input.
 * 2. Convert both strings to the same case (e.g., lowercase) to ensure the
 *    comparison is case-insensitive. The `std::transform` function is used
 *    for this purpose.
 * 3. Compare the two modified strings using standard string comparison operators.
 *    In C++, these operators perform a lexicographical comparison.
 * 4. If the strings are equal, print 0.
 * 5. If the first string is lexicographically greater than the second, print 1.
 * 6. Otherwise (the first is smaller), print -1.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. This is due
 * to the `transform` operations and the final comparison, which all take
 * linear time.
 * Space Complexity: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string input1, input2;
  cin >> input1;
  cin >> input2;

  /**
   * @brief Convert both strings to lowercase for case-insensitive comparison.
   * `std::transform` applies the `::tolower` function to every character
   * in the input strings, modifying them in-place.
   */
  transform(input1.begin(), input1.end(), input1.begin(), ::tolower);
  transform(input2.begin(), input2.end(), input2.begin(), ::tolower);

  /**
   * @brief Perform lexicographical comparison and print the result.
   * The standard string operators (==, >) provide the required
   * lexicographical comparison on the now-lowercase strings.
   */
  if (input1 == input2) {
    cout << "0" << endl;
  } else if (input1 > input2) {
    cout << "1" << endl;
  } else {
    cout << "-1" << endl;
  }

  return 0;
}
