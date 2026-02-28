/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A, "Petya and Strings".
 * @details This program performs a case-insensitive lexicographical comparison of two strings.
 * It reads two strings, converts them to the same case (lowercase) on-the-fly, and
 * determines if the first string is less than, greater than, or equal to the second.
 * @uthor chavodon
 * @link https://codeforces.com/problemset/problem/112/A
 *
 * @algorithm Single-Pass Lexicographical Comparison
 * 1. Read two strings, `a` and `b`.
 * 2. Iterate from the beginning to the end of the strings with a single loop.
 * 3. In each iteration, convert the characters at the current index from both
 *    strings to lowercase.
 * 4. Compare the two lowercase characters.
 * 5. If a difference is found, print -1 or 1 accordingly, set a flag, and break the loop.
 * 6. If the loop completes without the flag being set, it means the strings are equal, so print 0.
 *
 * @complexity
 * - Time: O(L), where L is the length of the strings. The program iterates through
 *   the strings at most once.
 * - Space: O(L), to store the two input strings.
 */

#include <iostream>
#include <string>
#include <cctype> // For tolower

using namespace std;

int main() {
  // a, b: The two strings to be compared.
  string a, b;
  // i: Loop counter.
  int i;
  // flag: Used to indicate if a difference between the strings has been found.
  // 0 means no difference found yet, 1 means a difference was found.
  int flag = 0;
  
  // Read the two input strings.
  cin >> a;
  cin >> b;

  /**
   * @brief Iterates through the strings, converting to lowercase and comparing.
   * This single loop is efficient as it avoids a separate pass for normalization.
   */
  // Invariant: `a` and `b` are lexicographically equal up to index `i-1`.
  for (i = 0; i < a.size(); i++) {
    // Convert characters at the current position to lowercase for case-insensitive comparison.
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);

    // Pre-condition: Check if the character in string `a` is less than in `b`.
    if (a[i] < b[i]) {
      cout << "-1";
      flag = 1; // Mark that a difference has been found.
      break;    // Exit the loop as the result is determined.
    } 
    // Pre-condition: Check if the character in string `a` is greater than in `b`.
    else if (a[i] > b[i]) {
      cout << "1";
      flag = 1; // Mark that a difference has been found.
      break;    // Exit the loop as the result is determined.
    }
  }

  /**
   * @brief Check for equality.
   * If the loop finished without the flag being set, it means no differences
   * were found, so the strings are equal.
   */
  if (flag == 0) {
    cout << "0";
  }
  
  return 0; // Indicate successful execution.
}
