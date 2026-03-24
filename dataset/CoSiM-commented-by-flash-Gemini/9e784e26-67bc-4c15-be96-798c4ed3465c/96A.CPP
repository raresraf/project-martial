/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem: "Football".
 *
 * Determines if a given binary string contains a substring of 7 or more consecutive identical characters ('0' or '1').
 * This identifies a "dangerous" situation in football (soccer).
 *
 * Algorithm: Linear scan with dedicated counters for consecutive '0's and '1's.
 * Time Complexity: O(L), where L is the length of the input string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the "Football" problem.
 * Reads a binary string and checks for 7 or more consecutive identical characters.
 * Outputs "YES" if found, "NO" otherwise.
 * @return 0 on successful execution (implicitly, as it's common in competitive programming).
 */
int main() {
  /**
   * Declares and initializes:
   * 'z': Counter for consecutive '0's.
   * 'o': Counter for consecutive '1's.
   * 's': The input binary string.
   * Reads the input string 's' from standard input.
   * Pre-condition: Standard input provides a binary string.
   */
  int z = 0, o = 1;
  string s;
  cin >> s;
  /**
   * Initializes loop index 'i'.
   *
   * Block Logic: Iterates through the input string 's' character by character.
   * It maintains two counters, 'z' for consecutive '0's and 'o' for consecutive '1's.
   * If 7 consecutive identical characters are found (either '0's or '1's), the loop terminates early.
   * Invariant: 'z' and 'o' accurately reflect the current consecutive counts.
   * Pre-condition: 's' is a null-terminated string.
   */
  int i = 0;
  while (s[i] != '\0') {
    /**
     * Block Logic: Checks the current character and updates the appropriate consecutive counter.
     * If the character is '0', increments 'z' and resets 'o'. If 'z' reaches 7, breaks.
     * If the character is '1', increments 'o' and resets 'z'. If 'o' reaches 7, breaks.
     */
    if (s[i] == '0') {
      z++;
      o = 0;
      if (z == 7)
        break;
    } else {
      o++;
      z = 0;
      if (o == 7)
        break;
    }
    i++;
  }
  /**
   * Block Logic: Outputs "YES" if 7 consecutive identical characters were found (either '0's or '1's), "NO" otherwise.
   * Post-condition: "YES" or "NO" is printed to standard output.
   */
  if (z == 7 || o == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
