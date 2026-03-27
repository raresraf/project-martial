/**
 * @file 266A.CPP
 * @brief This program counts the number of times consecutive characters in a given string are identical.
 *        Essentially, it counts how many characters need to be removed to make all adjacent characters distinct.
 *
 * Algorithm:
 * 1. Read an integer 'n' (the length of the string) and then the string 's'.
 * 2. Initialize a counter 'c' to 0.
 * 3. Iterate through the string from the first character up to the second-to-last character.
 * 4. In each iteration, compare the current character `s[i]` with the next character `s[i+1]`.
 * 5. If they are identical, increment the counter 'c'.
 * 6. Print the final count 'c'.
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
  // Block Logic: Declares an integer 'n' to store the length of the string.
  int n;
  // Block Logic: Reads the integer 'n' from standard input.
  cin >> n;
  // Block Logic: Declares a string 's' to store the input sequence of characters.
  string s;
  // Block Logic: Reads the string 's' from standard input.
  cin >> s;
  // Block Logic: Initializes a counter 'c' to keep track of consecutive identical characters.
  int c = 0;
  /**
   * Block Logic: Iterates through the string to compare adjacent characters.
   * The loop runs from the first character up to the second-to-last character (n-1)
   * to ensure a valid s[i+1] access.
   *
   * Pre-condition: The string 's' has been populated and its length is 'n'.
   * Invariant: At each iteration, 'c' accumulates the count of identical adjacent characters found so far.
   */
  for (int i = 0; i < n - 1; i++) {
    // Block Logic: Compares the current character with the next character in the string.
    if (s[i] == s[i + 1])
      // Block Logic: Increments the counter if the adjacent characters are identical.
      c++;
  }
  // Block Logic: Prints the total count of consecutive identical characters found.
  cout << c;
  return 0;
}
