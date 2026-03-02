#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem from Codeforces (Problem 96A).
 * @details This program determines if a football lineup is "dangerous," meaning at least
 * seven consecutive players are from the same team.
 *
 * Algorithm:
 * This solution iterates through the string of players, using two separate counters
 * to track the current consecutive runs of players from team '0' (`team1`) and team '1' (`team2`).
 * When a player from one team is encountered, its corresponding counter is incremented,
 * and the counter for the opposing team is reset to zero. After each player is processed,
 * the program checks if either counter has reached 7. If one has, a "dangerous" situation
 * is confirmed, "YES" is printed, and the program terminates. If the entire string is
 * processed without any counter reaching 7, the situation is not dangerous, and "NO" is printed.
 *
 * Time Complexity: O(L), where L is the length of the player string, due to the single pass.
 * Space Complexity: O(L) to store the input string.
 */
int main(void) {
  string players;

  // `team1` counts consecutive '0's, `team2` counts consecutive '1's.
  int team1 = 0, team2 = 0;

  // `danger` is a flag to indicate if a dangerous situation was found.
  int danger = 0;

  cin >> players;

  /**
   * @brief Iterate through the string of players to find a dangerous run.
   */
  for (int i = 0; i < players.length(); i++) {
    if (players[i] == '0') {
      // Increment the counter for team '0' and reset the counter for team '1'.
      team1++;
      team2 = 0;
    } else {
      // Increment the counter for team '1' and reset the counter for team '0'.
      team2++;
      team1 = 0;
    }

    // Check if either team has a run of 7 consecutive players.
    if (team1 == 7 || team2 == 7) {
      cout << "YES" << endl;
      danger = 1;
      break;
    }
  }

  // If the loop completes without finding a dangerous situation.
  if (!danger)
    cout << "NO" << endl;

  return 0;
}
