/**
 * @file 96A.CPP
 * @brief "Football" - Checks for a dangerous situation in player positions.
 *
 * This program reads a string representing a line of football players,
 * where each player is represented by either a '0' or a '1'.
 * It determines if there is a "dangerous" situation, which is defined as
 * having seven or more consecutive players of the same team (i.e., seven
 * consecutive '0's or seven consecutive '1's).
 *
 * @algorithm
 * 1. Read the input string representing the line of players.
 * 2. Initialize a counter `stat` to 1, which will track the number of
 *    consecutive identical characters.
 * 3. Iterate through the string from the second character to the end.
 * 4. For each character, compare it with the previous character.
 *    a. If they are the same, increment `stat`.
 *    b. If they are different, reset `stat` to 1.
 * 5. After incrementing, check if `stat` has reached 7.
 *    a. If it has, a dangerous situation is found. Print "YES", set a flag
 *       to indicate that "YES" has been printed, and break the loop.
 * 6. After the loop, if the "YES" flag has not been set, it means no
 *    dangerous situation was found. Print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, as it
 * requires a single pass through the string. `strlen` is called once before the loop in some compilers or on each iteration in others, but the overall complexity remains linear.
 * Space Complexity: O(L) to store the input string.
 */

#include <stdio.h>
#include <string.h>

int main() {
  // `string` will store the sequence of players, with a max length of 100.
  char string[101];
  
  // `stat` counts the number of consecutive identical players. Initialized to 1 for the first player.
  int stat = 1;
  
  // `printed` is a flag to check if "YES" has already been printed. 0 means no, 1 means yes.
  int printed = 0;
  
  // Read the player sequence from standard input.
  scanf("%s", string);
  
  /**
   * @brief Loop to check for 7 consecutive identical characters.
   *
   * It starts from the second character (index 1) to compare with the previous one.
   */
  for (int i = 1; i < strlen(string); i++) {
    // Invariant: At the start of the iteration, `stat` holds the count of the
    // consecutive block ending at `string[i-1]`.
    if (string[i] == string[i - 1]) {
      // If the current character is the same as the previous one, increment the counter.
      stat++;
      
      // Check if a dangerous situation has occurred.
      if (stat == 7) {
        printf("YES\n");
        printed = 1; // Set the flag to indicate we've found the answer.
        break;       // Exit the loop as the condition is met.
      }
    } else {
      // If the character changes, reset the consecutive counter.
      stat = 1;
    }
  }
  
  // If the loop finished without the 'printed' flag being set, no dangerous situation was found.
  if (printed == 0)
    printf("NO\n");
    
  return 0;
}
