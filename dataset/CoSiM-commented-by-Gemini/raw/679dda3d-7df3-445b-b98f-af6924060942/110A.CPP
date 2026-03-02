#include <bits/stdc++.h>
using namespace std;

/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem from Codeforces (Problem 110A).
 * @details This program determines if a given number is "nearly lucky". A number is
 * considered nearly lucky if the count of its "lucky digits" ('4' and '7') is
 * itself a lucky number. The only single-digit lucky numbers are 4 and 7.
 *
 * Algorithm:
 * The input number is read as a string to accommodate potentially large values. The
 * program then iterates through each character of the string. It maintains a counter
 * that is incremented every time a character is either '4' or '7'. After checking all
 * digits, the program examines the final count. If the count is 4 or 7, the original
 * number is nearly lucky, and the program prints "YES". Otherwise, it prints "NO".
 *
 * Time Complexity: O(L), where L is the number of digits in the input number. This is
 * because the program performs a single pass through the string representation of the number.
 * Space Complexity: O(L) to store the number as a string.
 */
int main() {
  string num;
  int counter = 0, i;

  // Read the number as a string.
  cin >> num;

  /**
   * @brief Count the occurrences of lucky digits ('4' and '7').
   * This loop iterates through each character of the input string.
   */
  for (i = 0; i < num.length(); i++) {
    if (num[i] == '4' || num[i] == '7')
      counter++;
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number.
   * The only possible lucky number counts from an 18-digit number are 4 and 7.
   */
  if (counter == 4 || counter == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
