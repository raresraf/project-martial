/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem (112A) from Codeforces.
 * @details This program compares two strings lexicographically, ignoring differences in letter case.
 *
 * @algorithm The solution involves three main steps:
 * 1. Read the two input strings.
 * 2. Convert both strings entirely to lowercase to ensure a case-insensitive comparison.
 * 3. Use the standard string comparison function to determine their lexicographical order.
 * 4. Print -1 if the first string is lexicographically smaller, 1 if it is larger, and 0 if they are equal.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. The process involves a single pass
 * to convert characters to lowercase and another pass for comparison, both of which are linear in time.
 * Space Complexity: O(L), for storing the two input strings.
 */
#include <iostream>
using namespace std;
int main() {
  string s, c;
  cin >> s >> c;
  int i;
  /**
   * @block
   * @brief Converts both strings to lowercase for case-insensitive comparison.
   * @pre 's' and 'c' contain the input strings.
   * @post All uppercase characters in 's' and 'c' are converted to lowercase.
   */
  for (i = 0; s[i] != '\0'; i++) {
    s[i] = tolower(s[i]);
    c[i] = tolower(c[i]);
  }
  /**
   * @block
   * @brief Performs lexicographical comparison and prints the result.
   * @pre 's' and 'c' are the lowercase versions of the input strings.
   * @post Prints -1, 0, or 1 based on the comparison result.
   */
  if (s.compare(c) < 0) {
    cout << "-1";
  } else if (s.compare(c) > 0) {
    cout << "1";
  } else
    cout << 0;
}
