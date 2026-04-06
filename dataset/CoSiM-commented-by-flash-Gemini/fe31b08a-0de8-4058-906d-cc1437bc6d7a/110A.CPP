/**
 * @file 110A.CPP
 * @brief Alternative solution for Codeforces Problem 110A - "Nearly Lucky Number".
 * This program determines if a given number is "nearly lucky" based on a specific
 * interpretation where the count of lucky digits must be either 4 or 7.
 *
 * @problem_statement
 * A number is considered "nearly lucky" if the total count of '4's and '7's
 * within its decimal representation is itself equal to 4 or 7.
 *
 * @algorithm
 * 1. Read the input number as a string (`holder`) to easily iterate through its digits.
 * 2. Initialize a counter `noluckey` to 0.
 * 3. Iterate through each character of the input string:
 *    a. If the character is '4' or '7', increment `noluckey`.
 * 4. After counting, check if `noluckey` is equal to 4 or 7.
 * 5. Print "YES" if `noluckey` is 4 or 7; otherwise, print "NO".
 *
 * Time Complexity: O(L), where L is the number of digits (length of the input string).
 *   The program iterates through the digits of the input number once.
 * Space Complexity: O(L), to store the input number as a string.
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Main function to implement the "Nearly Lucky Number" logic.
   * Reads the input number as a string, counts the occurrences of lucky digits
   * ('4' and '7'), and then determines if this count itself is a lucky number
   * (specifically, if it is 4 or 7).
   * @return 0 on successful execution.
   */
  int length; ///< @var length Stores the length of the input number string.
  int noluckey = 0; ///< @var noluckey Counts the occurrences of lucky digits ('4' or '7').
  string holder; ///< @var holder Stores the input number as a string for easy digit access.

  // Read the input number as a string.
  cin >> holder;
  length = holder.length(); // Get the length of the input string.

  // Using a C-style array `almostlucky` is unnecessary as `holder.at(i)` can be used directly.
  // This line can be removed without affecting logic.
  char almostlucky[length]; 

  /**
   * @brief Iterates through each character (digit) of the input string to count lucky digits.
   * @invariant `i` increments from 0 to `length-1`, processing each digit.
   * `noluckey` accumulates the count of '4's and '7's found so far.
   * @post `noluckey` holds the total count of lucky digits in the input number.
   */
  for (int i = 0; i < length; i++) {
    almostlucky[i] = holder.at(i); // Assign current character to C-style array (redundant for logic).
    /**
     * @brief Checks if the current character is a lucky digit ('4' or '7').
     */
    if (holder.at(i) == '4' || holder.at(i) == '7') {
      noluckey++; // Increment the count of lucky digits.
    }
  }

  /**
   * @brief Determines and prints the final result based on the count of lucky digits.
   * @details If the count of lucky digits (`noluckey`) is exactly 4 or 7, the number
   * is considered "nearly lucky" and "YES" is printed. Otherwise, "NO" is printed.
   */
  if (noluckey == 4 || noluckey == 7) {
    cout << "YES"; // The count of lucky digits is itself a lucky number (4 or 7).
  } else {
    cout << "NO"; // The count of lucky digits is not 4 or 7.
  }
  return 0;
}