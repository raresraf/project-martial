/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem determining gender based on username's distinct characters.
 * @details This program reads a username as a string, counts the number of unique characters
 * in it, and then determines whether to "CHAT WITH HER!" (even number of distinct
 * characters) or "IGNORE HIM!" (odd number). This is a classic problem from platforms
 * like Codeforces.
 *
 * @algorithm Frequency counting using an array.
 * @complexity
 * Time complexity: O(N), where N is the length of the input string, for iterating through the string once.
 * Space complexity: O(1), as it uses a fixed-size array (26 for English alphabet).
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function to execute the solution.
 */
int main(int argc, char const *argv[]) {
  /**
   * @param arr An array to act as a frequency map for the lowercase English alphabet.
   * @param s The input username string.
   */
  int arr[26] = {0};
  string s;
  cin >> s;
  /**
   * @brief This loop iterates through the input string to mark the presence of each character.
   * Pre-condition: 's' contains the username.
   * Invariant: 'arr' marks the characters that have been seen so far.
   */
  for (int i = 0; s[i]; i++) {
    // Mark the character as present in the frequency array.
    // 97 is the ASCII value for 'a'.
    arr[s[i] - 97] = 1;
  }
  int sum = 0;
  /**
   * @brief This loop calculates the number of distinct characters.
   * Pre-condition: 'arr' is populated with 1s for characters present in the string.
   * Invariant: 'sum' accumulates the count of distinct characters.
   */
  for (int i = 0; i < 26; i++) {
    sum += arr[i];
  }
  /**
   * @brief Check if the number of distinct characters is even or odd to determine the output.
   */
  if (sum % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}