/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem: "Football".
 *
 * Determines if a given string contains a substring of 7 or more consecutive identical characters.
 * This identifies a "dangerous" situation in football (soccer).
 *
 * Algorithm: Linear scan with a consecutive character counter.
 * Time Complexity: O(L), where L is the length of the input string.
 */
#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Main function to solve the "Football" problem.
 * Reads a string and checks for 7 or more consecutive identical characters.
 * Outputs "YES" if found, "NO" otherwise.
 * @return 0 on successful execution.
 */
int main() {
  /**
   * Declares variables:
   * 'a': The input string representing player positions.
   * 'flag1', 'flag2', 'c2': Unused variables (potential remnants or incomplete logic).
   * 'c1': Counter for consecutive identical characters, initialized to 1.
   * Reads the input string 'a' from standard input.
   * Pre-condition: Standard input provides a string.
   */
  string a;
  int flag1 = 0, flag2 = 0, c1 = 1, c2 = 0;
  cin >> a;
  /**
   * Block Logic: Iterates through the input string to count consecutive identical characters.
   * If 7 consecutive identical characters are found, the loop terminates early.
   * Invariant: 'c1' stores the current count of consecutive identical characters.
   * Pre-condition: 'a' contains the input string.
   */
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == a[i + 1]) {
      c1++;
      if (c1 == 7)
        break;
    } else
      c1 = 1;
  }
  /**
   * Block Logic: Outputs "YES" if 7 consecutive identical characters were found, "NO" otherwise.
   * Post-condition: "YES" or "NO" is printed to standard output.
   */
  if (c1 == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
