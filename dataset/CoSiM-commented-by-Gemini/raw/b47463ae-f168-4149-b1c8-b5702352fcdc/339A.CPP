/**
 * @file 339A.CPP
 * @brief Sorts a string representing a sum of numbers.
 *
 * The input is a string representing a sum of numbers, where the numbers
 * are '1', '2', and '3', separated by '+'. The program rearranges the
 * numbers in non-decreasing order. For example, "3+1+2" becomes "1+2+3".
 *
 * @algorithm Counting Sort
 * @details The program uses a counting sort approach. It first counts the
 * occurrences of each number ('1', '2', '3'). Then, it reconstructs the
 * string by appending the numbers in sorted order, separated by '+'.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string.
 * The program iterates through the string once to count the numbers and then
 * a few more times to build the result string, which is still linear in N.
 * Space Complexity: O(N) to store the resulting string.
 */
#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  string s;
  getline(cin, s);
  int c1 = 0, c2 = 0, c3 = 0;
  
  // A special case for a single digit input.
  if (s.length() < 3) {
    cout << s[0];
    return 0; // Exit after handling the simple case.
  }

  /**
   * @brief Count the occurrences of each number.
   * This loop iterates through the input string and increments counters
   * for each of the numbers '1', '2', and '3'.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      c1++;
    if (s[i] == '2')
      c2++;
    if (s[i] == '3')
      c3++;
  }

  /**
   * @brief Reconstruct the string in sorted order.
   * This series of if-else blocks builds the result string by first
   * appending all '1's, then all '2's, and finally all '3's, with
   * '+' signs in between.
   */
  if (c1 > 0) {
    string res = "1";
    for (int i = 1; i < c1; i++)
      res = res + "+1";
    for (int i = 0; i < c2; i++)
      res = res + "+2";
    for (int i = 0; i < c3; i++)
      res = res + "+3";
    cout << res;
  } else if (c2 > 0) {
    string res = "2";
    for (int i = 1; i < c2; i++)
      res = res + "+2";
    for (int i = 0; i < c3; i++)
      res = res + "+3";
    cout << res;
  } else {
    string res = "3";
    for (int i = 1; i < c3; i++)
      res = res + "+3";
    cout << res;
  }

  return 0;
}
