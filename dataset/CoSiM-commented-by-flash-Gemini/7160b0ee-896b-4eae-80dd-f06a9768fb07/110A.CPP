/**
 * @file 110A.CPP
 * @brief Solution to Codeforces Problem 110A: "Nearly Lucky Number".
 * Determines if a given number is "nearly lucky". A number is considered nearly lucky
 * if the count of 'lucky digits' (which are '4' and '7') within it is itself a lucky number.
 * Algorithm: Iterate through the digits of the input number (represented as a string),
 * count occurrences of '4' and '7'. Then, check if this count is either 4 or 7.
 * Time Complexity: O(L), where L is the number of digits in the input number.
 * Space Complexity: O(L) to store the input number as a string.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream and cstring for strlen.
using namespace std;

/**
 * @brief Main function to solve the "Nearly Lucky Number" problem.
 * Reads a number as a string, counts its lucky digits, and prints "YES" or "NO".
 */
int main() {
  char s[100]; // Character array to store the input number as a string (max 99 digits + null terminator).
  int i;       // Loop counter.
  int cn = 0;  // Counter for lucky digits ('4' or '7').
  int ln;      // Stores the length of the input string.
  
  scanf("%s", &s);     // Reads the input number as a string.
  ln = strlen(s);      // Calculates the length of the string.

  /**
   * Block Logic: Iterates through each character (digit) of the input string.
   * Pre-condition: 's' contains the input number, 'ln' its length, 'cn' is 0.
   * Invariant: 'cn' accurately accumulates the number of lucky digits encountered so far.
   */
  for (i = 0; i < ln; i++) {
    /**
     * Block Logic: Checks if the current digit is a '4' or a '7'.
     * If it is, increments the lucky digit counter.
     */
    if (s[i] == '4' || s[i] == '7') {
      cn++; // Increment count of lucky digits.
    }
  }

  /**
   * Block Logic: Determines if the count of lucky digits ('cn') is itself a lucky number.
   * Prints "YES" if 'cn' is 4 or 7, otherwise prints "NO".
   */
  if (cn == 4 || cn == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
