/**
 * @file 236A.CPP
 * @brief Determines a response based on the parity of distinct characters in a username.
 *
 * This program reads a username string and counts the number of unique
 * characters it contains. If the count of distinct characters is odd,
 * it outputs "IGNORE HIM!". If the count is even, it outputs
 * "CHAT WITH HER!". This logic is often found in competitive programming
 * problems to introduce simple string manipulation and conditional logic.
 *
 * Algorithm: Character frequency analysis using a set for uniqueness, followed by a parity check.
 * Time Complexity: O(L) where L is the length of the input `userName`.
 *                  The `getline` operation takes O(L). The loop iterates L times,
 *                  and `set::insert` for a character (assuming a limited character set like ASCII)
 *                  takes amortized O(1) time.
 * Space Complexity: O(K) where K is the number of distinct characters in the input string.
 *                   In the worst case for ASCII, K is at most 256.
 */

#include <iostream>
#include <set>

using namespace std;

int main() {
  /**
   * @brief Main function to execute the distinct character parity check.
   *
   * Initializes data structures, reads the username, processes its characters
   * to find distinct ones, and then outputs a message based on the parity
   * of the distinct character count.
   */
  // @param userName Stores the input string (username).
  string userName;
  // @param distinctCharacters A set to store unique characters encountered in the username.
  set<char> distinctCharacters;

  // Reads the entire line of input into the `userName` variable.
  getline(cin, userName);
  /**
   * @brief Iterates through each character of the username to populate the set of distinct characters.
   * Invariant: `distinctCharacters` always contains all unique characters from the substring `userName[0...i-1]`.
   */
  for (int i = 0; i < userName.size(); i++) {
    distinctCharacters.insert(userName[i]);
  }

  /**
   * @brief Determines the output message based on the parity of distinct character count.
   *
   * If the number of distinct characters is odd, the message "IGNORE HIM!" is printed.
   * If the number is even, "CHAT WITH HER!" is printed.
   */
  if (distinctCharacters.size() % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";

  return 0;
}
