/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to lowercase, and then compares them
 * character by character. It is intended to determine which string is lexicographically
 * smaller, but it has a logical flaw: it does not stop at the first differing character,
 * so the final result only reflects the comparison of the last characters if they differ.
 *
 * @b Algorithm:
 * 1.  Read two strings from standard input.
 * 2.  Convert both strings to lowercase using `std::transform`.
 * 3.  Iterate through the strings from beginning to end.
 * 4.  In each iteration, compare the characters at the current position. Update a result
 *     variable if one character is greater than the other.
 * 5.  After the loop, print the final value of the result variable.
 *
 * @b Time_Complexity: O(L), where L is the length of the strings.
 * @b Space_Complexity: O(L) to store the two input strings.
 * @b Note: The logic is flawed. For a correct lexicographical comparison, the loop
 * should terminate as soon as the first difference is found.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  // Read two lines of input into two strings.
  getline(cin, s1);
  getline(cin, s2);
  // Convert both strings to lowercase for case-insensitive comparison.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  int res = 0;
  /**
   * @brief This loop compares the two strings character by character.
   *
   * Pre-condition: s1 and s2 are the lowercase input strings.
   * Flaw: The loop does not break after finding the first difference, so the `res`
   * variable will be overwritten in subsequent iterations if other characters differ.
   * The final output will depend only on the relationship of the last differing characters.
   */
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] > s2[i])
      res = 1;
    else if (s1[i] < s2[i])
      res = -1;
  }
  // Output the final result of the comparison.
  cout << res;
  return 0;
}