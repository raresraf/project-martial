/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces.
 * @details The problem defines a "lucky number" as a positive integer containing
 * only the digits '4' and '7'. A "nearly lucky number" is one where the
 * count of lucky digits ('4' or '7') within it is itself a lucky number.
 * This program determines if a given number is nearly lucky.
 *
 * @algorithm
 * 1. Read the entire number as a string to facilitate digit-by-digit inspection.
 * 2. Iterate through each character of the string.
 * 3. Maintain a counter for the number of times '4' or '7' appears.
 * 4. After iterating through the string, check if the final count is a lucky
 *    number. Given the constraints (input number has at most 18 digits), the
 *    count can only be 4 or 7 to be a lucky number.
 * 5. Print "YES" if the count is 4 or 7, otherwise print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the number of digits in the input number,
 * as the algorithm involves a single pass through the string.
 * Space Complexity: O(L) to store the number as a string.
 */

/// Problem Name: Nearly Lucky Number
/// Problem Link: https://codeforces.com/problemset/problem/110/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  string s; // To store the input number as a string.
  cin >> s;

  // `lucky_digit_count` will store the number of occurrences of '4' and '7'.
  int lucky_digit_count = 0;

  /**
   * @block
   * @brief Count the number of lucky digits ('4' and '7') in the input number.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '4' || s[i] == '7')
      lucky_digit_count++;
  }

  /**
   * @block
   * @brief Check if the count of lucky digits is itself a lucky number.
   * @details For the given constraints, the only possible "lucky" counts are 4 and 7.
   */
  if (lucky_digit_count == 4 || lucky_digit_count == 7)
    cout << "YES" << '\n';
  else
    cout << "NO" << '\n';
    
  return 0;
}