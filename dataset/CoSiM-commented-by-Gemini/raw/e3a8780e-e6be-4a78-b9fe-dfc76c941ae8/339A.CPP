/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 * @description This program rearranges a string representing a sum of numbers (1, 2, or 3)
 *              into non-decreasing order. For example, "3+1+2" becomes "1+2+3".
 *
 * Algorithm: This solution utilizes a Counting Sort approach, which is highly efficient
 *            because the set of numbers is very small and known (only 1, 2, and 3).
 * 1. Read the input string.
 * 2. Iterate through the string once to count the occurrences of '1', '2', and '3'.
 * 3. Reconstruct the output string by first appending all the '1's, then all the '2's,
 *    and finally all the '3's, separated by '+' signs.
 * 4. Print the final string, removing the trailing '+' that results from the
 *    construction process.
 *
 * Time Complexity: O(N), where N is the length of the input string. This is because
 *                  the counting and reconstruction phases are both linear passes.
 * Space Complexity: O(N) to store the input and output strings.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string input, output;
  cin >> input;
  int count1 = 0, count2 = 0, count3 = 0;

  /**
   * Block Logic (Counting Phase): Iterate through the input string to count the
   * frequency of each number ('1', '2', '3').
   */
  for (int i = 0; i < input.length(); i++) {
    if (input[i] == '1')
      count1++;
    else if (input[i] == '2')
      count2++;
    else if (input[i] == '3')
      count3++;
  }

  /**
   * Block Logic (Reconstruction Phase): Build the output string based on the counts.
   * This implicitly sorts the numbers. The condition `countX > 0` is redundant but harmless.
   */
  int i;
  for (i = 0; i < count1 && count1 > 0; i++)
    output += "1+";
  for (i = 0; i < count2 && count2 > 0; i++)
    output += "2+";
  for (i = 0; i < count3 && count3 > 0; i++)
    output += "3+";
  
  // Print the final string, removing the extraneous trailing '+'.
  cout << output.substr(0, output.length() - 1);
  return 0;
}
