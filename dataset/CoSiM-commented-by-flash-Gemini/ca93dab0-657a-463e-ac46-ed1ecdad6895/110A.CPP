/**
 * @file 110A.CPP
 * @brief Determines if a given number is "nearly lucky".
 *   A number is considered "nearly lucky" if the count of '4's and '7's in its decimal representation
 *   is either 4 or 7.
 * * Algorithm: Digit Count and Comparison.
 *   The program reads a number as a string, iterates through its digits, and counts
 *   occurrences of '4' and '7'. It then checks if this count itself is a lucky number (4 or 7).
 * Time Complexity: O(L) where L is the number of digits in the input number (length of the string).
 * Space Complexity: O(L) for storing the input number as a string.
 */

#include <iostream>
using namespace std;

/**
 * @brief Checks if a given number (represented as a string) is "nearly lucky".
 * @param x The string representation of the number to check.
 * Functional Utility: Counts lucky digits and determines the "nearly lucky" status.
 */
void luck_check(string x) {
  unsigned long c = 0; // Counter for lucky digits ('4' or '7').
  /**
   * Block Logic: Iterates through each digit of the input string.
   * Invariant: 'c' accumulates the count of lucky digits encountered so far.
   */
  for (unsigned long i = 0; i < x.size(); i++) {
    /**
     * Condition: Checks if the current digit is '7' or '4'.
     * If true, increments the lucky digit counter.
     */
    if (x[i] == '7' || x[i] == '4') {
      c++;
    }
  }
  /**
   * Condition: Determines if the count of lucky digits itself is a lucky number.
   * If 'c' is 4 or 7, the original number is "nearly lucky".
   */
  if (c == 4 || c == 7) {
    cout << "YES";
  } else
    cout << "NO";
}

/**
 * @brief Main function to read the number and initiate the luck check.
 * Functional Utility: Handles input/output and calls the 'luck_check' function.
 */
int main() {
  string num; // Stores the input number as a string.
  cin >> num; // Reads the number.
  luck_check(num); // Calls the function to check if 'num' is nearly lucky.
  // Redundant and incorrect declaration. Left as is due to "Zero Code Mutation" mandate.
  void luck_check();
  return 0; // Indicates successful program execution.
}
