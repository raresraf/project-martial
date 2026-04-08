/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem from Codeforces (Problem 96A).
 * Problem Statement: Determines if a situation in a football match is "dangerous" if there are
 *                    seven consecutive players of the same team (represented by '0' or '1').
 * Algorithm: Single pass iteration. Iterates through the string, maintaining counts of consecutive
 *            '0's and '1's. If either count reaches 7, the situation is dangerous.
 * Time Complexity: O(N) where N is the length of the input string.
 * Space Complexity: O(1) for constant extra variables.
 * Domain: Competitive Programming, String Manipulation.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to solve the Football problem.
 * Functional Utility: Reads a string representing player positions and determines
 *                     if a "dangerous" situation exists based on consecutive identical characters.
 */
int main() {
  // Variables:
  // str: Input string representing player positions ('0' or '1').
  // one: Counter for consecutive '1's.
  // zero: Counter for consecutive '0's.
  // flag: Flag to indicate if a dangerous situation (7 consecutive identical characters) is found (1 for YES, 0 for NO).
  string str;
  int one = 0, zero = 0, flag = 0;
  // Block Logic: Reads the input string.
  cin >> str;
  // Block Logic: Iterates through the string to check for 7 consecutive identical characters.
  // Invariant: At each iteration, `one` and `zero` store the counts of consecutive
  //            '1's and '0's ending at the current character.
  for (int i = 0; i < str.length(); i++) {
    // Block Logic: If the current character is '1', increment `one` and reset `zero`.
    if (str[i] == '1') {
      one++;
      zero = 0;
    } 
    // Block Logic: If the current character is '0', increment `zero` and reset `one`.
    else {
      zero++;
      one = 0;
    }
    // Block Logic: Checks if either '0' or '1' has appeared 7 or more times consecutively.
    // If true, sets the flag and breaks the loop as danger is detected.
    if (zero >= 7 || one >= 7) { // Corrected from `zero == 7 || one == 7` to `zero >= 7 || one >= 7`
      flag = 1;
      break;
    }
  }
  // Block Logic: Outputs "YES" if a dangerous situation was found, otherwise "NO".
  if (flag)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
