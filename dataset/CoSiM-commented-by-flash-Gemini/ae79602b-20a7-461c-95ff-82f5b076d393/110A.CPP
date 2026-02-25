/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces (Problem 110A).
 *        This program determines if a given number is "nearly lucky". A number is
 *        nearly lucky if the count of "lucky digits" (4 and 7) within it is itself
 *        a lucky number (i.e., the count is either 4 or 7).
 * Algorithm: Digit Counting and Conditional Check
 * Time Complexity: O(L), where L is the length of the input string (number).
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, including iostream and cstring.
using namespace std;

/**
 * @brief Main function to check if an input number is a "nearly lucky number".
 * @return 0 if the program executes successfully.
 */
int main() {
  int i;         // Loop counter.
  int flag = 0;  // Counter for "lucky digits" (4s and 7s).
  char str[100]; // Character array to store the input number as a string.
  
  // Read the input number as a string.
  cin >> str;

  /**
   * Block Logic: Iterates through each character (digit) of the input string.
   * Invariant: 'flag' counts the occurrences of '4' and '7' encountered so far.
   */
  for (i = 0; str[i] != '\0'; i++) { // Loop until the null terminator of the string is reached.
    // Check if the current digit is either '4' or '7'.
    if (str[i] == '4' || str[i] == '7')
      flag++; // If it's a lucky digit, increment the flag counter.
  }
  
  /**
   * Block Logic: Determines if the count of lucky digits is itself a lucky number.
   * Pre-condition: 'flag' holds the total count of lucky digits in the input number.
   * Post-condition: Prints "YES" if 'flag' is 4 or 7, otherwise prints "NO".
   */
  if (flag == 4 || flag == 7) // Check if the count of lucky digits is 4 or 7.
    cout << "YES" << endl;    // If so, the number is "nearly lucky".

  else
    cout << "NO" << endl;     // Otherwise, it is not.
  
  return 0; // Indicate successful program termination.
}
