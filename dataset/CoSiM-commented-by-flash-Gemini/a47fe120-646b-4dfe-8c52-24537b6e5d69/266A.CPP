/**
 * @file 266A.CPP
 * @brief This program counts the number of times consecutive characters in a given string are identical.
 *        Essentially, it counts how many characters need to be removed to make all adjacent characters distinct.
 *
 * Algorithm:
 * 1. Read an integer 'n' (the length of the string) and then the string 's'.
 * 2. Initialize a counter 'm' to 0.
 * 3. Iterate through the string from the first character up to the second-to-last character.
 * 4. In each iteration, compare the current character `s[i]` with the next character `s[i+1]`.
 * 5. If they are identical, increment the counter 'm'. This is concisely done using a ternary operator.
 * 6. Print the final count 'm'.
 *
 * Time Complexity: O(N) where N is the length of the string, as the program iterates through the string once.
 * Space Complexity: O(N) to store the input string.
 */

#include <iostream>
#include <string>
using namespace std;
/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of string data, performs the
 * counting of consecutive identical characters, and outputs the final count.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares an integer 'n' for the string length (though not explicitly used for loop bounds here),
  //              and 'm' as a counter for identical consecutive characters, initialized to 0.
  int n, m = 0;
  // Block Logic: Declares a string 's' to store the input sequence of characters.
  string s;
  // Block Logic: Reads the integer 'n' (string length) from standard input.
  cin >> n;
  // Block Logic: Reads the string 's' from standard input.
  cin >> s;
  /**
   * Block Logic: Iterates through the string to compare adjacent characters.
   * The loop runs from the first character up to the second-to-last character (s.length() - 1)
   * to ensure a valid `s[i+1]` access.
   *
   * Invariant: At each iteration, 'm' accumulates the count of identical adjacent characters found so far.
   */
  for (int i = 0; i < s.length(); i++)
    // Inline: Uses a ternary operator to concisely increment 'm' if adjacent characters are identical.
    // If s[i] equals s[i+1], 'm' is incremented; otherwise, 'm' remains unchanged.
    m = s[i] == s[i + 1] ? m + 1 : m;
  // Block Logic: Prints the total count of consecutive identical characters found, followed by a newline.
  cout << m << endl;
  return 0;
}
