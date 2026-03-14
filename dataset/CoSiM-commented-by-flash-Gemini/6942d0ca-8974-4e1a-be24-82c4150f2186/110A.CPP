/**
 * @6942d0ca-8974-4e1a-be24-82c4150f2186/110A.CPP
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
#include <cmath>    // For mathematical functions (unused in this specific logic).
#include <cstdio>   // For C-style input/output functions (unused, cin/cout are used).
#include <cstring>  // For C-style string functions (unused, std::string is used).
#include <iostream> // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

typedef long long ll; // @brief Defines 'll' as a shorthand for 'long long' for convenience.

/**
 * @brief Main function to determine if a number is "nearly lucky".
 * Reads a number as a string, counts its lucky digits, and checks if the count is a lucky number.
 * @return 0 implicitly, as there is no explicit return statement at the end of main.
 */
int main() {
  string s; // Declares a string variable to store the input number.
  cin >> s; // Reads the number as a string from standard input.
  
  ll len = s.length(); // Stores the length of the input number string.
  ll count = 0; // Counts the occurrences of lucky digits (4s and 7s).
  
  /**
   * Block Logic: Iterates through each character (digit) of the input string `s`.
   * Increments `count` if the digit is '4' or '7'.
   * Invariant: `count` holds the total count of lucky digits in the number.
   */
  for (int i = 0; i < len; i++) { // Loops through each character of the string.
    // Inline: Checks if the current character is '7' or '4'.
    if (s[i] == '7' || s[i] == '4') { 
      count++; // Increments the lucky digit counter.
    }
  }
  
  /**
   * Block Logic: Checks if the count of lucky digits (`count`) is itself a lucky number (4 or 7).
   * Pre-condition: `count` contains the total count of lucky digits from the input number.
   * Invariant: Prints "YES" if `count` is 4 or 7, "NO" otherwise.
   */
  if (count == 4 || count == 7) { // Checks if the count of lucky digits is 4 or 7.
    cout << "YES" << endl; // Prints "YES" if it's a nearly lucky number.
  } else {
    cout << "NO" << endl; // Prints "NO" otherwise.
  }
}
