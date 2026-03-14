/**
 * @6957a801-5ebc-48fb-9207-24cf6c3865e9/96A.CPP
 * @brief This program solves Codeforces problem 96A, "Football".
 * It analyzes a sequence of players (represented by '0's and '1's in a string)
 * to detect a "dangerous situation". A dangerous situation occurs if there are
 * 7 or more consecutive players of the same team (i.e., "0000000" or "1111111"
 * appears as a substring).
 *
 * Algorithm: Linear scan with two counters. It iterates through the string,
 * maintaining separate counts for consecutive '0's and '1's. If either count
 * reaches 7, a "dangerous situation" is detected.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <algorithm> // For general algorithms (unused in this specific logic).
#include <iostream>  // For standard input/output operations (cin, cout).
#include <stdio.h>   // For C-style input/output (unused in this specific logic).
#include <stdlib.h>  // For general utilities (unused in this specific logic).
#include <string.h>  // For C-style string functions (unused, std::string is used).
#include <tgmath.h>  // For generic math functions (unused in this specific logic).

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to detect a "dangerous situation" in a football game.
 * Reads a string representing player sequence and checks for 7 consecutive same-team players.
 * @return 0 on successful execution.
 */
int main(void) {
  string players; // Declares a string variable to store the input sequence of players.
  cin >> players; // Reads the player sequence from standard input.

  int team1 = 0; // Counts consecutive '0's (representing Team 1).
  int team2 = 0; // Counts consecutive '1's (representing Team 2).
  int danger = 0; // Flag to indicate if a dangerous situation is found (0 for NO, 1 for YES).

  /**
   * Block Logic: Iterates through the input string to count consecutive identical characters.
   * If a '0' is encountered, `team1` is incremented and `team2` is reset.
   * If a '1' is encountered, `team2` is incremented and `team1` is reset.
   * If either `team1` or `team2` reaches 7, a dangerous situation is found.
   * Invariant: `team1` and `team2` hold the current streak of identical characters.
   */
  for (int i = 0; i < players.length(); i++) { // Loops through each character of the string.
    if (players[i] == '0') { // If the current player is '0'.
      team1++; // Increment count for team 1.
      team2 = 0; // Reset count for team 2.
    } else { // If the current player is '1'.
      team2++; // Increment count for team 2.
      team1 = 0; // Reset count for team 1.
    }

    // Pre-condition: Checks if either count for consecutive identical characters has reached 7.
    if (team1 == 7 || team2 == 7) { 
      cout << "YES" << endl; // Prints "YES".
      danger = 1; // Sets the flag to indicate a dangerous situation.
      break;      // Exits the loop early.
    }
  }

  // Block Logic: If the loop completed without finding a dangerous situation (`danger` is still 0).
  if (!danger) 
    cout << "NO" << endl; // Prints "NO".

  return 0; // Indicates successful program termination.
}
