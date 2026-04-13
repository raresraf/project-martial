/**
 * @file 110A.CPP
 * @brief This program determines if a given number is "almost lucky".
 * A number is considered lucky if all its digits are either '4' or '7'.
 * A number is "almost lucky" if the count of its lucky digits (4s and 7s)
 * is itself a lucky number.
 * Algorithm: Digit counting followed by string parsing to check luckiness of the count.
 * Time Complexity: O(L), where L is the number of digits in the input string.
 * Space Complexity: O(L) for storing the input string and the string representation of the count.
 */
#include <iostream>
#include <sstream>
using namespace std;
// Macro to redefine main function, a common practice in competitive programming.
#define soads_game int main()

soads_game {
  string s; // Stores the input number as a string.
  // Reads the input number as a string.
  cin >> s;
  int con = 0; // Counter for lucky digits ('4' or '7').
  bool r = true; // Flag to determine if the count of lucky digits is itself a lucky number.

  /**
   * Block Logic: Iterates through the input string to count the occurrences of lucky digits ('4' or '7').
   * Invariant: 'con' accurately reflects the number of lucky digits encountered up to the current index.
   */
  for (int i = 0; i < s.length(); i++) {
    if ((s[i] == '4') || (s[i] == '7'))
      con++;
  }

  stringstream gone; // Used to convert the integer count 'con' into a string.
  gone << con;      // Puts the integer 'con' into the stringstream.
  string n;         // Stores the string representation of the lucky digit count.
  gone >> n;        // Extracts the string from the stringstream into 'n'.

  /**
   * Block Logic: Checks if the number of lucky digits (represented as string 'n') is itself a lucky number.
   * A number is lucky if all its digits are '4' or '7'.
   * Invariant: 'r' remains true as long as all digits of 'n' are lucky digits.
   */
  for (int j = 0; j < n.length(); j++) {
    if ((n[j] != '4') && (n[j] != '7'))
      r = false; // If any digit is not '4' or '7', the count is not a lucky number.
  }

  /**
   * Conditional Logic: Outputs "YES" if the original number is "almost lucky", otherwise "NO".
   * An "almost lucky" number has a lucky digit count that is itself a lucky number.
   */
  if (r)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  // Standard program exit.
  return 0;
}

