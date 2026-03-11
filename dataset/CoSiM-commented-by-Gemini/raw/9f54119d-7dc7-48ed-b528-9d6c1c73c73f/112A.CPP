/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem.
 *
 * This program performs a case-insensitive lexicographical comparison of two input strings.
 * It outputs -1 if the first string is smaller, 1 if it's larger, and 0 if they are equal.
 *
 * Algorithm:
 * 1. Read two strings.
 * 2. Convert both strings to the same case (lowercase in this implementation).
 * 3. Compare the strings character by character.
 * 4. The first point of difference determines the lexicographical order.
 * 5. If no differences are found after checking all characters, the strings are equal.
 *
 * Time Complexity: O(L), where L is the length of the strings.
 * Space Complexity: O(L) to store the two strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string f, s; // The first and second strings for comparison.

  // Loop to handle multiple test cases, common in competitive programming.
  while (cin >> f >> s) {
    
    /**
     * @brief Convert both strings to lowercase for case-insensitive comparison.
     * Logic: Iterates through each character and converts uppercase letters to lowercase
     *        by adjusting their ASCII values.
     */
    for (int i = 0; i < f.size(); i++) {
      if (f[i] >= 'A' && f[i] <= 'Z')
        f[i] = f[i] - 'A' + 'a'; // Convert to lowercase.

      if (s[i] >= 'A' && s[i] <= 'Z')
        s[i] = s[i] - 'A' + 'a'; // Convert to lowercase.
    }

    /**
     * @brief Perform lexicographical comparison.
     * Pre-condition: Both strings 'f' and 's' are now in lowercase.
     * Invariant: Iterates through the strings until a difference is found or the end is reached.
     */
    for (int i = 0; i < s.size(); i++) {
      if (f[i] > s[i]) {
        cout << "1
";
        return 0; // First string is lexicographically greater.
      }

      if (s[i] > f[i]) {
        cout << "-1
";
        return 0; // Second string is lexicographically greater.
      }
    }
    
    // If the loop completes without returning, the strings are equal.
    cout << "0
";
  }
  return 0;
}
