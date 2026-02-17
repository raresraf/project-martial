/**
 * @file 96A.CPP
 * @brief Solution to a programming problem that checks for a "dangerous" situation in a football game.
 * @details The problem requires determining if there is a sequence of 7 or more identical consecutive characters in a given string, which represents players of two teams. '0' might represent one team and '1' the other. A sequence of 7 identical characters is considered a "dangerous" situation.
 *
 * @algorithm Single-pass linear scan with a running counter for consecutive characters.
 * @time_complexity O(L), where L is the length of the input string. The algorithm iterates through the string once.
 * @space_complexity O(L) to store the input string.
 */
#include <iostream>
using namespace std;

int main() {
  int n, total = 1; // 'total' tracks the current count of consecutive identical characters. 'n' is unused.
  char t; // Stores the character from the previous position for comparison.
  string kata; // The input string representing the sequence of players.
  cin >> kata;

  /**
   * Block Logic: Iterates through the string to check for consecutive identical characters.
   * Invariant: 'total' holds the length of the current run of identical characters ending at index i-1.
   */
  for (int i = 0; i < kata.length(); i++) {
    // Pre-condition: This check avoids accessing an invalid index on the first iteration.
    if (i > 0) {
      // Logic: If the current character is the same as the previous one, increment the counter.
      if (kata[i] == t) {
        total++;
      } else {
        // If the character changes, reset the counter.
        total = 1;
      }
      // Condition: If a sequence of 7 is found, the condition is met, and we can exit early.
      if (total == 7) {
        break;
      }
    }
    // Update the previous character for the next iteration.
    t = kata[i];
  }
  
  // Output: Based on whether a sequence of 7 was found.
  if (total == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}