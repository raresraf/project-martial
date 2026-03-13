/**
 * @file 110A.CPP
 * @brief Determines if a number is "nearly lucky".
 *
 * A number is considered "nearly lucky" if the count of its "lucky digits"
 * (4 and 7) is a lucky number itself (i.e., the count is 4 or 7).
 * This program reads an integer as a string, counts the occurrences of the
 * characters '4' and '7', and then checks if this count is 4 or 7.
 *
 * @see https://codeforces.com/problemset/problem/110/A
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, flag = 0, n;
  char str[100];
  cin >> str;

  /**
   * @brief Counts the lucky digits in the input string.
   *
   * This loop iterates through each character of the input string `str`.
   * It increments `flag` for each occurrence of '4' or '7'.
   */
  for (i = 0; str[i] != '\0'; i++) {
    if (str[i] == '4' || str[i] == '7')
      flag++;
  }
  /**
   * @brief Checks if the count of lucky digits is itself a lucky number.
   *
   * If the final count (`flag`) is 4 or 7, the number is "nearly lucky".
   */
  if (flag == 4 || flag == 7)
    cout << "YES" << endl;

  else
    cout << "NO" << endl;
}
