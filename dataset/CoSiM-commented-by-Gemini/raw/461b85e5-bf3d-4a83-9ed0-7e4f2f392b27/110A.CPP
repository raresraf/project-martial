/**
 * @file 110A.CPP
 * @brief Solution for a programming contest problem involving "lucky numbers".
 * @details The problem defines a "nearly lucky number". A number is nearly lucky
 * if the count of its "lucky digits" ('4' and '7') is itself a lucky number.
 * The lucky numbers are positive integers whose decimal representation contains only digits 4 and 7.
 *
 * Algorithm:
 * 1. Read the input number as a string.
 * 2. Iterate through the digits of the string and count the occurrences of '4' and '7'.
 * 3. Check if the resulting count is a lucky number. Given the typical constraints
 *    for such problems (input number up to 10^18, having at most 19 digits), the
 *    count of lucky digits can be at most 19. The only lucky numbers within this
 *    range are 4 and 7.
 * 4. If the count is 4 or 7, the number is "nearly lucky", so print "YES".
 * 5. Otherwise, print "NO".
 *
 * Time Complexity: O(L) where L is the number of digits in the input number,
 * as it requires a single pass through the string.
 * Space Complexity: O(L) to store the input number as a string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int cou = 0;
  char s[100];
  cin >> s;
  
  /**
   * @block
   * @description This loop counts the number of lucky digits ('4' or '7')
   * in the input number string.
   */
  for (int i = 0; i < strlen(s); i++) {
    if (s[i] == '4' || s[i] == '7')
      cou++;
  }
  
  /**
   * @block
   * @description Checks if the count of lucky digits is itself a lucky number.
   * For the problem's constraints, the only possible lucky number counts are 4 and 7.
   */
  if (cou == 4 || cou == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}