/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem (Codeforces 110A).
 *
 * This program determines if a given number is a "nearly lucky number".
 * A number is nearly lucky if the count of its "lucky digits" (4 and 7)
 * is itself a lucky number (i.e., either 4 or 7).
 *
 * Algorithm: Digit counting and conditional check. The program iterates
 * through the digits of the input number (represented as a string), counts
 * occurrences of '4' and '7', and then checks if this count is 4 or 7.
 * Time Complexity: O(L) where L is the number of digits in the input number (string length).
 * Space Complexity: O(L) to store the input number string.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Main function to determine if an input number is a "nearly lucky number".
 *
 * Reads a number as a string, counts the occurrences of lucky digits (4 and 7),
 * and then checks if this count is itself a lucky number.
 *
 * @return void (standard competitive programming practice; implicitly returns 0 on successful exit).
 */
int main() {
  // Declare an integer 'cou' to count lucky digits.
  int cou = 0;
  // Declare a character array 's' to store the input number as a string.
  char s[100];
  // Read the input number into 's'.
  cin >> s;
  /**
   * @brief Block Logic: Iterates through the input string to count lucky digits.
   *
   * Pre-condition: 's' contains the input number.
   * Invariant: 'cou' accumulates the count of '4's and '7's found so far.
   */
  for (int i = 0; i < strlen(s); i++) {
    // Check if the current character is '4' or '7'.
    if (s[i] == '4' || s[i] == '7')
      cou++; // Increment lucky digit count.
  }
  /**
   * @brief Block Logic: Determines if the count of lucky digits is itself lucky.
   *
   * Pre-condition: 'cou' holds the total count of lucky digits in the input number.
   * Invariant: Prints "YES" if 'cou' is 4 or 7, "NO" otherwise.
   */
  if (cou == 4 || cou == 7) {
    cout << "YES"; // The number is "nearly lucky".
  } else {
    cout << "NO";  // The number is not "nearly lucky".
  }
}
