/**
 * @file 110A.CPP
 * @brief Solution to a competitive programming problem: "Nearly Lucky Number".
 * Determines if a given number (represented as a string) is a "nearly lucky" number.
 * A number is nearly lucky if the count of 'lucky digits' (4 or 7) within it
 * is itself a lucky number (4 or 7).
 * Algorithm: Linear scan to count lucky digits, then check the count.
 * Time Complexity: O(L) where L is the number of digits in the input number.
 * Space Complexity: O(L) for storing the input string, or O(1) if input is streamed character by character.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the Nearly Lucky Number problem.
 * Reads a number as a string, counts the occurrences of '4' and '7' digits.
 * Based on this count, it determines and prints whether the number is "nearly lucky".
 */
int main() {
  string s;
  cin >> s;
  int count = 0; // Counter for lucky digits (4s and 7s).

  /**
   * Block Logic: Iterates through each character (digit) of the input string.
   * Pre-condition: 's' contains the input number as a string. 'count' is 0.
   * Invariant: 'count' accumulates the number of lucky digits encountered.
   */
  for (int i = 0; i < s.length(); i++) {
    /**
     * Block Logic: Checks if the current digit is a '4' or a '7'.
     * If it is, increment the lucky digit counter.
     */
    if (s[i] == '4' || s[i] == '7') {
      count++;
    }
  }

  /**
   * Block Logic: Determines if the count of lucky digits is itself a lucky number.
   * If 'count' is 4 or 7, the original number is "nearly lucky".
   */
  if (count == 4 || count == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
