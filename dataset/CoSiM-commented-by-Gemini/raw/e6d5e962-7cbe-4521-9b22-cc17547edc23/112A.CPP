/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces (112A).
 * @details This program performs a case-insensitive lexicographical comparison of two input strings.
 * It returns -1 if the first string is lexicographically smaller than the second,
 * 1 if it's greater, and 0 if they are equal.
 *
 * @section
 * Algorithm:
 * 1. Read two strings, s1 and s2.
 * 2. Convert all characters in both strings to lowercase to ensure the comparison is case-insensitive.
 * 3. Use standard string comparison to determine the lexicographical order of the modified strings.
 * 4. Print -1, 1, or 0 based on the comparison result.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(L), where L is the length of the strings. This is determined by the
 *   loop for character conversion and the final string comparison.
 * - Space Complexity: O(L), to store the two input strings.
 */
#include <iostream>
#include <string>
#include <cctype> // Required for tolower

using namespace std;

int main() {
  string s1, s2;

  // Read the two strings to be compared.
  cin >> s1 >> s2;

  int length = s1.length();

  /**
   * Block Logic: Normalize both strings to lowercase.
   * This step is crucial for performing a case-insensitive comparison as required
   * by the problem.
   */
  for (int i = 0; i < length; i++) {
    s1[i] = static_cast<char>(tolower(s1[i]));
    s2[i] = static_cast<char>(tolower(s2[i]));
  }

  /**
   * Block Logic: Perform lexicographical comparison.
   * C++ string operators directly compare strings lexicographically. After
   * normalization, this comparison meets the problem's requirements.
   */
  if (s1 < s2) {
    cout << "-1" << endl;
  }

  else if (s1 > s2) {
    cout << "1" << endl;
  }

  else {
    cout << "0" << endl;
  }

  return 0;
}
