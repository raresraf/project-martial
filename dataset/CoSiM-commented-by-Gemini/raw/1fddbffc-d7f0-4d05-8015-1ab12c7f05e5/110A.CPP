/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" competitive programming problem.
 *
 * @details This program determines if a given number is "nearly lucky". A lucky
 * number is defined as a positive integer containing only the digits 4 and 7.
 * A number is considered "nearly lucky" if the count of lucky digits (4s and 7s)
 * within it is itself a lucky number.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the input number as a string to handle large values.
 * 2. Count the occurrences of the characters '4' and '7' in the input string.
 *    Let this count be `value`.
 * 3. Convert the integer `value` into its own string representation.
 * 4. Check if this new string (representing the count) is composed exclusively
 *    of the characters '4' and '7'.
 * 5. If it is, the original number is "nearly lucky", and the program prints "YES".
 * 6. Otherwise, it prints "NO".
 *
 * @section complexity
 * Time Complexity: O(L), where L is the number of digits in the input number.
 * This is dominated by the initial pass to count lucky digits.
 * Space Complexity: O(L), primarily to store the input string.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s1;
  // This loop structure is often used for multiple test cases, but for this
  // specific problem, it will only execute once.
  while (cin >> s1) {
    int len1, i, value = 0, a;
    len1 = s1.length();

    // Block Logic: Count the number of lucky digits ('4' or '7') in the input string.
    for (i = 0; i < len1; i++) {
      if (s1[i] == '4' || s1[i] == '7')
        value = value + 1;
    }

    // This check is redundant. If value is 0, the subsequent logic correctly
    // determines it's not a lucky number.
    if (value == 0)
      cout << "NO" << endl;
    else {
      // Block Logic: Check if the count ('value') is itself a lucky number.
      string res = "";
      // Convert the integer 'value' to its string representation 'res'.
      for (i = value; i != 0; i = i / 10) {
        res.push_back(i % 10 + '0');
      }
      reverse(res.begin(), res.end());
      
      a = 0;
      // Count the number of lucky digits in the new string 'res'.
      for (i = 0; i < res.length(); i++) {
        if (res[i] == '4' || res[i] == '7')
          a++;
      }

      // If the count of lucky digits 'a' is equal to the length of the string 'res',
      // it means all digits in 'value' were lucky.
      if (a == res.length())
        cout << "YES" << endl;
      else
        cout << "NO" << endl;
    }
  }
  return 0;
}
