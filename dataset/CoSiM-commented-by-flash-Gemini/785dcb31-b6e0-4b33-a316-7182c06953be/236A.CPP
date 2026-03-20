/**
 * @file 236A.CPP
 * @brief Another variant solution to the Codeforces problem 236A - Boy or Girl.
 *        The goal is to determine the output based on the parity of distinct characters in a username.
 * Algorithm:
 *   1. Reads a string (username).
 *   2. Sorts the characters of the string to group identical characters together.
 *   3. Attempts to count the number of "distinct" characters by iterating through the sorted string
 *      and incrementing a counter when a character differs from the next.
 *      Note: A logical error exists in the counting loop where 'c++' is effectively always executed
 *      unless 'name[i] == name[i+1]' is true, due to an unintended semicolon after the 'if' condition.
 *      This results in 'c' counting character transitions differently from a true distinct character count.
 *   4. The final output ("CHAT WITH HER!" or "IGNORE HIM!") is determined by the parity of 'c',
 *      which is incremented once more before the modulo operation (`c++ % 2`).
 * Time Complexity: O(N log N) due to sorting, where N is the length of the input string.
 * Space Complexity: O(N) for storing the string (or O(1) if in-place sort and string modifications are considered).
 */
#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace std;
int main() {
  /**
   * @brief Main function to solve the "Boy or Girl" problem.
   *
   * Reads a username, sorts its characters, and then attempts to count
   * the number of distinct characters for parity checking.
   * Due to an identified logical bug, the character counting mechanism
   * does not accurately represent the distinct character count.
   * The final output is based on the parity of this potentially incorrect count.
   * @return int Returns 0 upon successful execution.
   */
  int c = 0;
  string name;
  cin >> name;
  // Block Logic: Sorts the characters of the input name string.
  // Pre-condition: 'name' contains the input username.
  // Post-condition: Characters in 'name' are arranged in non-decreasing order.
  sort(name.begin(), name.end());

  // Block Logic: Iterates through the sorted string to count "distinct" characters.
  // Invariant: 'c' attempts to count distinct characters by checking adjacent elements.
  //            However, due to a misplaced semicolon, 'else c++;' effectively increments 'c'
  //            whenever name[i] != name[i+1] is false (i.e., name[i] == name[i+1]),
  //            and also in the 'else' branch when name[i] != name[i+1] is true.
  //            This leads to 'c' not correctly representing the number of unique characters.
  for (int i = 0; i < (int)name.length(); i++) {
    if (name[i] == name[i + 1])
      ; // This semicolon makes the 'if' body empty.
    else
      c++; // This 'else' clause is paired with the 'if' condition from the previous line.
  }

  // Block Logic: Determines the output message based on the parity of 'c'.
  // Pre-condition: 'c' holds a count derived from character comparisons (affected by the bug).
  // Post-condition: Prints "CHAT WITH HER!" if (c+1) is even, "IGNORE HIM!" if (c+1) is odd.
  // Note: 'c++ % 2' first uses the value of 'c' in the modulo operation, then increments 'c'.
  if (c++ % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
