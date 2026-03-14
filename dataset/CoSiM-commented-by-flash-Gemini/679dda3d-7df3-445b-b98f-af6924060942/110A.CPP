/**
 * @679dda3d-7df3-445b-b98f-af6924060942/110A.CPP
 * @brief This program solves Codeforces problem 110A, "Nearly Lucky Number".
 * A number is considered "lucky" if it contains only the digits 4 and 7.
 * A number is "nearly lucky" if the count of its lucky digits (4s and 7s)
 * is itself a lucky number (i.e., 4 or 7). The program reads a number
 * (as a string) and determines if it is nearly lucky.
 *
 * Algorithm: Digit counting and conditional checking. Iterates through the
 * digits of the input number (as a string), counts occurrences of '4' and '7',
 * then checks if this count is 4 or 7.
 * Time Complexity: O(L) where L is the number of digits in the input number.
 * Space Complexity: O(L) for storing the input number as a string.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream> and <string>.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to determine if a number is "nearly lucky".
 * Reads a number as a string, counts its lucky digits, and checks if the count is a lucky number.
 * @return 0 on successful execution.
 */
int main() {
  string num; // Declares a string variable to store the input number.
  int counter = 0; // Counts the occurrences of lucky digits (4s and 7s).
  int i; // Loop counter.
  
  cin >> num; // Reads the number as a string from standard input.
  
  /**
   * Block Logic: Iterates through each character (digit) of the input string `num`.
   * Increments `counter` if the digit is '4' or '7'.
   * Invariant: `counter` holds the total count of lucky digits in the number.
   */
  for (i = 0; i < num.length(); i++) { // Loops through each character of the string.
    // Inline: Checks if the current character is '4' or '7'.
    if (num[i] == '4' || num[i] == '7') 
      counter++; // Increments the lucky digit counter.
  }
  
  /**
   * Block Logic: Checks if the count of lucky digits (`counter`) is itself a lucky number (4 or 7).
   * Pre-condition: `counter` contains the total count of lucky digits from the input number.
   * Invariant: Prints "YES" if `counter` is 4 or 7, "NO" otherwise.
   */
  if (counter == 4 || counter == 7) // Checks if the count of lucky digits is 4 or 7.
    cout << "YES" << endl; // Prints "YES" if it's a nearly lucky number.
  else
    cout << "NO" << endl; // Prints "NO" otherwise.
  
  return 0; // Indicates successful program termination.
}
