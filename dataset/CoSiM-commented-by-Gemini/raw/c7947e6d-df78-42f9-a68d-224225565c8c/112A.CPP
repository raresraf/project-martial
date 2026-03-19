/**
 * @file raw/c7947e6d-df78-42f9-a68d-224225565c8c/112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings and compares them as if they were all uppercase.
 * It outputs -1 if the first string is lexicographically smaller, 1 if it is
 * greater, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings into C-style character arrays.
 * 2. Iterate through both strings simultaneously.
 * 3. In each iteration, convert the characters at the current position from both strings to
 *    uppercase using ASCII value manipulation.
 * 4. Compare the now-uppercase characters.
 * 5. If they are different, print -1 or 1 based on the comparison result and terminate.
 * 6. If the loop completes without finding any differences, it means the strings are equal, so print 0.
 *
 * @time_complexity O(L), where L is the length of the strings, as it requires a
 *                  single pass.
 * @space_complexity O(L_max), for storing the strings in character arrays.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  char ch[101], ch2[101];
  int i, sum = 0, sum1 = 0, sum2 = 0; // Unused sum variables
  cin >> ch >> ch2;
  /**
   * @block
   * Iterates through the strings, performing case-insensitive comparison
   * character by character.
   */
  for (i = 0; i < strlen(ch); i++) {
    // Inline: Convert current character of the first string to uppercase.
    if (ch[i] >= 97 && ch[i] <= 122) {
      ch[i] -= 32;
    }
    // Inline: Convert current character of the second string to uppercase.
    if (ch2[i] >= 97 && ch2[i] <= 122) {
      ch2[i] -= 32;
    }
    // Perform the comparison.
    if (ch[i] > ch2[i]) {
      cout << "1" << endl;
      break; // Exit as soon as a difference is found.
    } else if (ch[i] < ch2[i]) {
      cout << "-1" << endl;
      break; // Exit as soon as a difference is found.
    }
  }
  /**
   * @block
   * This check is executed after the loop. If the loop completed without a
   * break, it means all characters were equal. In that case, `i` will equal
   * `strlen(ch)`, and this condition will compare the null terminators,
   * resulting in a correct output of 0.
   */
  if (ch[i] == ch2[i])
    cout << "0" << endl;
}
