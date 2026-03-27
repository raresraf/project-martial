/**
 * @file 266A.CPP
 * @brief This program counts the number of times consecutive characters in a given string are identical.
 *        Essentially, it counts how many characters need to be removed to make all adjacent characters distinct.
 *
 * Algorithm:
 * 1. Read an integer 'n' (the length of the string) and then the string 's'.
 * 2. Initialize a counter 'count' to 0.
 * 3. Iterate through the string starting from the second character (`i = 1`).
 * 4. In each iteration, compare the current character `s[i]` with the previous character `s[i-1]`.
 * 5. If they are identical, increment the counter 'count'.
 * 6. Print the final count 'count'.
 *
 * Time Complexity: O(N) where N is the length of the string, as the program iterates through the string once.
 * Space Complexity: O(N) to store the input string.
 */

#include <iostream>
using namespace std;
/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of string data, performs the
 * counting of consecutive identical characters, and outputs the final count.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares an integer 'n' for the string length,
  //              and 'count' as a counter for identical consecutive characters, initialized to 0.
  int n, count = 0;
  // Block Logic: Declares a string 's' to store the input sequence of characters.
  string s;
  // Block Logic: Reads the integer 'n' (string length) and the string 's' from standard input.
  cin >> n >> s;
  /**
   * Block Logic: Iterates through the string, starting from the second character,
   * to compare each character with its immediate predecessor.
   *
   * Invariant: At each iteration, 'count' accumulates the number of times a character
   * is identical to the one before it. The loop ensures all adjacent pairs are checked.
   */
  for (int i = 1; i < n; i++) {
    // Block Logic: Compares the current character `s[i]` with the previous character `s[i-1]`.
    if (s[i] == s[i - 1])
      // Block Logic: Increments the counter if the adjacent characters are identical.
      count++;
  }
  // Block Logic: Prints the total count of consecutive identical characters found, followed by a newline.
  cout << count << endl;
  return 0;
}
