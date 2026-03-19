/**
 * @file 96A.CPP
 * @brief Solution to Codeforces Problem 96A: "Football".
 *
 * This problem determines if a situation in a football match is "dangerous".
 * A situation is dangerous if there are 7 or more consecutive players of the same team.
 * The input is a string representing the sequence of players (0 for one team, 1 for the other).
 *
 * Algorithm: Iterative scan for consecutive identical characters.
 * The algorithm iterates through the input string, keeping a count of consecutive
 * identical characters. If the count reaches 7, the situation is dangerous.
 *
 * Time Complexity: O(N), where N is the length of the input string, as it involves
 *                  a single pass through the string.
 * Space Complexity: O(1), as only a few variables are used to store counts and flags.
 */
#include <stdio.h> // Required for input/output operations (scanf, printf).
#include <string.h> // Required for string manipulation functions (strlen).

int main() {
  char string[101]; // Declares a character array (string) to store the input player sequence. Max length 100 + null terminator.
  int stat = 1; // Counter for consecutive identical players. Initializes to 1 for the first character.
  int printed = 0; // Flag to indicate if "YES" has been printed, preventing duplicate output.
  scanf("%s", string); // Reads the input string representing player positions.
  
  /**
   * Block Logic: Iterates through the input string to find sequences of identical players.
   * Pre-condition: The 'string' array contains the player sequence, 'stat' is 1, 'printed' is 0.
   * Invariant: 'stat' tracks the count of consecutive identical characters ending at the current index 'i'.
   */
  for (int i = 1; i < strlen(string); i++) {
    // Block Logic: Checks if the current player is the same as the previous one.
    if (string[i] == string[i - 1]) {
      stat++; // Increments the counter for consecutive identical players.
      // Block Logic: If 7 or more consecutive identical players are found, the situation is dangerous.
      if (stat == 7) {
        printf("YES\n"); // Prints "YES" to indicate a dangerous situation.
        printed = 1; // Sets the flag to prevent further output.
        break; // Exits the loop as the condition is met.
      }
    } else {
      stat = 1; // Resets the counter if the current player is different from the previous one.
    }
  }
  // Block Logic: If no dangerous situation was found throughout the string, prints "NO".
  if (printed == 0)
    printf("NO\n");
  return 0; // Indicates successful program execution.
}
