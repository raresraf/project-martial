/**
 * @file 110A.CPP
 * @brief This program determines if a given string contains a "nearly lucky" number.
 * A number is considered "lucky" if it consists only of the digits 4 and 7.
 * A string is "nearly lucky" if the count of lucky digits (4s and 7s) within it
 * is itself a lucky number.
 *
 * Algorithm:
 * 1. Count the occurrences of '4' and '7' in the input string.
 * 2. Check if this count is a "lucky" number (i.e., composed only of digits 4 and 7).
 * Time Complexity: O(N) where N is the length of the input string, due to single pass for counting lucky digits.
 * Space Complexity: O(1) for storing the count.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Checks if a given integer is a "lucky" number.
 * A number is lucky if all its digits are either 4 or 7.
 * @param x The integer to check.
 * @pre x is a non-negative integer.
 * @post Returns true if x consists only of digits 4 and 7, false otherwise.
 */
bool lucky(int x) {
  // Block Logic: Initial check for numbers less than 4, which cannot be lucky.
  // Invariant: Numbers less than 4 are trivially not lucky.
  if (x < 4)
    return false;
  else {
    // Block Logic: Iterates through each digit of the number to validate its luckiness.
    // Invariant: At each iteration, x is the remaining part of the number to be checked.
    while (x > 0) {
      // Block Logic: Checks if the last digit is neither 4 nor 7.
      // Pre-condition: x > 0.
      // Invariant: The current last digit is being evaluated for luckiness.
      if (x % 10 != 4 && x % 10 != 7)
        return false;
      x /= 10;
    }
    // Functional Utility: If all digits are 4 or 7, the number is lucky.
    return true;
  }
}

/**
 * @brief Solves the "nearly lucky" number problem.
 * Reads a string, counts lucky digits, and determines if the count is a lucky number.
 * @pre Input string 's' contains only digits.
 * @post Prints "YES" if the count of lucky digits in 's' is a lucky number, "NO" otherwise.
 */
void solve() {
  string s;
  cin >> s;
  int dem = 0; // Functional Utility: Stores the count of lucky digits (4s and 7s) in the string.
  int n = s.length();

  // Block Logic: Iterates through the input string to count lucky digits.
  // Invariant: 'dem' accurately reflects the count of '4's and '7's encountered so far.
  for (int i = 0; i < n; i++) {
    // Block Logic: Increments 'dem' if the current character is a lucky digit.
    // Pre-condition: s[i] is a character from the input string.
    if (s[i] == '4' || s[i] == '7')
      dem++;
    // else
    // {
    //   cout<<"NO";
    //  exit(0);
    //}
  }
  // Block Logic: Determines and prints the final result based on whether 'dem' is a lucky number.
  // Pre-condition: 'dem' contains the total count of lucky digits in the input string.
  if (lucky(dem))
    cout << "YES";
  else
    cout << "NO";
}

/**
 * @brief Main function of the program.
 * Orchestrates the execution of the solution.
 * @return 0 upon successful execution.
 */
int main() {
  solve();

  return 0;
}
