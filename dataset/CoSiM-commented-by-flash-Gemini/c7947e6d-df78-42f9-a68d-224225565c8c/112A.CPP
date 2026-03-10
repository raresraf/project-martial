/**
 * @file 112A.CPP
 * @brief This program solves the "Petya and Strings" problem (Codeforces 112A).
 *   It compares two input strings lexicographically, ignoring case.
 *   Problem description: https://codeforces.com/problemset/problem/112/A
 * * Algorithm: Case-insensitive lexicographical comparison by converting to uppercase.
 *   1. Reads two input strings as C-style character arrays.
 *   2. Iterates through both arrays, converting any lowercase characters to uppercase in-place.
 *   3. Compares the characters of the two strings (now effectively uppercase) one by one.
 *   4. The first differing characters determine the lexicographical order, and the loop breaks early.
 *   5. If the loop completes without finding differences, the strings are considered equal.
 * Time Complexity: O(L) where L is the length of the strings (assuming they are of similar length),
 *   due to iterating through the strings once for case conversion and comparison.
 * Space Complexity: O(L) to store the input strings.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <cstring> (for strlen).
using namespace std;

/**
 * @brief Main function to perform case-insensitive string comparison.
 * Functional Utility: Reads two strings, converts them to uppercase manually,
 *                     compares them character by character to find lexicographical order,
 *                     and prints the comparison result (-1, 0, or 1).
 */
int main() {
  char ch[101], ch2[101]; // C-style character arrays to store the two input strings.
  int i;                  // Loop counter.
  // `sum`, `sum1`, `sum2`: Declared but unused variables, likely remnants from a template or previous attempts.
  int sum = 0, sum1 = 0, sum2 = 0;
  // Block Logic: Reads the two input strings into `ch` and `ch2`.
  cin >> ch >> ch2;
  /**
   * Block Logic: Iterates through the strings, converting characters to uppercase and performing comparison.
   * Pre-condition: `ch` and `ch2` are populated with input strings.
   * Invariant: Characters are converted to uppercase in-place. The loop breaks early upon finding a difference.
   * Inline: `strlen(ch)` computes the length of `ch`. Assumes both strings are of equal length.
   */
  for (i = 0; i < strlen(ch); i++) {
    // Conditional Logic: Converts lowercase characters in `ch[i]` to uppercase using ASCII arithmetic.
    if (ch[i] >= 97 && ch[i] <= 122) { // ASCII values: 'a'=97, 'z'=122.
      ch[i] -= 32;                     // Subtracts 32 to convert to uppercase.
    }
    // Conditional Logic: Converts lowercase characters in `ch2[i]` to uppercase using ASCII arithmetic.
    if (ch2[i] >= 97 && ch2[i] <= 122) { // ASCII values: 'a'=97, 'z'=122.
      ch2[i] -= 32;                      // Subtracts 32 to convert to uppercase.
    }

    /**
     * Conditional Logic: Compares the (now uppercase) characters at the current position `i`.
     * If `ch[i]` is greater than `ch2[i]`, `ch` is lexicographically larger.
     */
    if (ch[i] > ch2[i]) {
      cout << "1" << endl; // Prints "1".
      break;               // Breaks loop as order is determined.
    }
    /**
     * Conditional Logic: If `ch[i]` is less than `ch2[i]`, `ch` is lexicographically smaller.
     */
    else if (ch[i] < ch2[i]) {
      cout << "-1" << endl; // Prints "-1".
      break;                // Breaks loop as order is determined.
    }
  }
  // Conditional Logic: If the loop completes without breaking (i.e., `ch[i] == ch2[i]` for all `i`),
  // it means the strings are equal. The loop `i` will be `strlen(ch)` at this point.
  // The condition `ch[i] == ch2[i]` will effectively compare their null terminators.
  if (ch[i] == ch2[i])
    cout << "0" << endl; // Prints "0".
}
