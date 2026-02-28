/**
 * @file 96A.CPP
 * @brief Checks for a "dangerous" situation in a string based on consecutive identical characters.
 *
 * This program reads a single string from standard input. It then iterates
 * through the string to find sequences of identical consecutive characters.
 * If any sequence consists of 7 or more identical characters, the situation
 * is deemed "dangerous", and the program outputs "YES". Otherwise, it outputs "NO".
 *
 * Algorithm: Single-pass string traversal with state tracking. It keeps a count
 *            of consecutive identical characters and resets the count when the
 *            character changes.
 * Time Complexity: O(L) where L is the length of the input string, as the program
 *                  iterates through the string once.
 * Space Complexity: O(1) beyond the storage required for the input string itself,
 *                   as only a few scalar variables are used for state tracking.
 */

#include <iostream>
using namespace std;

int main() {
  /**
   * @brief Main function to detect "dangerous" sequences in an input string.
   *
   * Initializes constants and variables for state tracking, reads the input
   * string, and then iterates through it to count consecutive identical characters.
   * Sets the output to "YES" if a dangerous sequence is found.
   */
  // @param dangerous The threshold for consecutive identical characters to be considered "dangerous".
  const int dangerous = 7;
  // @param line Stores the input string from the user.
  string line;
  // Reads the entire line of input from standard input into `line`.
  getline(cin, line);
  // @param state Stores the character being currently counted for consecutive occurrences.
  //              Initialized to -1 (an invalid char value) to ensure the first character always triggers a state change.
  int state = -1;
  // @param runningSum Counts the number of consecutive occurrences of the current `state` character.
  int runningSum = 0;
  // @param output Stores the final result, initialized to "NO".
  string output = "NO";
  /**
   * @brief Loop to iterate through each character of the input string to find consecutive sequences.
   * Invariant: `runningSum` accurately reflects the count of consecutive `state` characters ending at `line[k-1]`.
   */
  for (int k = 0; k < line.size(); k++) {
    // @brief Checks if the current character is different from the previously tracked character.
    if (line[k] != state) {
      state = line[k];    // Update the state to the new character.
      runningSum = 0;     // Reset the count of consecutive occurrences.
    }
    ++runningSum; // Increment the count for the current consecutive character.
    // @brief Checks if the `runningSum` has reached or exceeded the `dangerous` threshold.
    if (runningSum >= dangerous) {
      output = "YES"; // Set output to "YES" if dangerous condition is met.
      break;          // No need to check further, break the loop.
    }
  }
  // @brief Prints the final result ("YES" or "NO") to standard output, followed by a newline.
  cout << output << endl;
}
