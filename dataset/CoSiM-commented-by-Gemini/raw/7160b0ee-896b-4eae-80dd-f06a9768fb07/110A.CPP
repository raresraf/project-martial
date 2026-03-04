/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A - "Nearly Lucky Number".
 *
 * This program checks if a given number is a "nearly lucky number". A nearly
 * lucky number is defined as a number where the count of its "lucky digits"
 * (which are 4 and 7) is itself a lucky number (i.e., the count is 4 or 7).
 * The input is read as a string to handle large numbers.
 *
 * @algorithm Digit counting and checking.
 * @time_complexity O(L), where L is the number of digits in the input number
 *                  (the length of the string), as it iterates through the
 *                  string once.
 * @space_complexity O(L) to store the input number as a string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  int i, cn = 0, ln;
  // Read the input number as a string.
  scanf("%s", &s);
  ln = strlen(s);

  /**
   * @brief Counts the occurrences of lucky digits (4 and 7).
   * This loop iterates through each character of the input string and
   * increments a counter `cn` if the character is either '4' or '7'.
   */
  for (i = 0; i < ln; i++) {
    if (s[i] == '4' || s[i] == '7')
      cn++;
  }

  /**
   * @brief Determines if the number is nearly lucky.
   * After counting the lucky digits, this check determines if the count
   * itself is a lucky number (4 or 7). If it is, the number is "nearly
   * lucky".
   */
  if (cn == 4 || cn == 7) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;

  return 0;
}