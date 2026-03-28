/**
 * @file 110A.CPP
 * @brief Determines if a number is "nearly lucky".
 *
 * A number is nearly lucky if the number of lucky digits ('4' and '7')
 * in it is a lucky number itself. This program counts the occurrences of '4'
 * and '7' in an input string, and then checks if that count is a lucky number.
 *
 * @b Algorithm:
 * 1. Count the number of '4' and '7' digits in the input string.
 * 2. Convert this count into a string.
 * 3. Check if the string representation of the count consists only of '4' and '7'.
 *
 * @b Time_Complexity: O(L + log(C)), where L is the length of the input string
 *    and C is the count of lucky digits. This is because it first iterates
 *    through the input string, and then iterates through the digits of the count.
 * @b Space_Complexity: O(L) for storing the input string.
 */
#include <iostream>
#include <sstream>
using namespace std;
#define soads_game int main()

soads_game {
  string s; // The input number as a string.
  cin >> s;
  int con = 0; // Counter for lucky digits ('4' or '7').
  bool r = true; // Flag to check if the count is a lucky number.

  /**
   * @brief First loop: count the lucky digits in the input string.
   */
  for (int i = 0; i < s.length(); i++) {
    if ((s[i] == '4') || (s[i] == '7'))
      con++;
  }

  // Convert the integer count to a string to check if it's a lucky number.
  stringstream gone;
  gone << con;
  string n;
  gone >> n;

  /**
   * @brief Second loop: check if the count of lucky digits is itself a lucky number.
   */
  for (int j = 0; j < n.length(); j++) {
    if ((n[j] != '4') && (n[j] != '7'))
      r = false;
  }

  if (r)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
