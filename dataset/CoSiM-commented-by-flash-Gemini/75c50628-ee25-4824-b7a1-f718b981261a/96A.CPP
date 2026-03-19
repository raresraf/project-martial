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
 * Space Complexity: O(N) due to std::string storage, or O(1) if input is considered a stream.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).

int main() {
  std::string s; // Stores the input string representing the sequence of players.
  std::cin >> s; // Reads the input string.
  char c = s[0], count = 1; // 'c' stores the current character being counted, 'count' tracks consecutive occurrences.
  
  /**
   * Block Logic: Iterates through the input string to find sequences of identical players.
   * Pre-condition: 's' contains the player sequence, 'c' is the first char, 'count' is 1.
   * Invariant: 'count' tracks the consecutive occurrences of character 'c'.
   */
  for (unsigned int i = 0; i < s.size(); ++i) {
    // Block Logic: Checks if the current character is different from the previously counted character.
    if (s[i] != c) {
      c = s[i]; // Updates the character being counted.
      count = 1; // Resets the consecutive count.
    } else if (s[i] == c) {
      ++count; // Increments the count if the character is the same.
    }
    // Block Logic: If 7 or more consecutive identical players are found, the situation is dangerous.
    if (count == 7) {
      std::cout << "YES\n"; // Prints "YES" to indicate a dangerous situation.
      return 0; // Exits the program as the condition is met.
    }
  }
  // Block Logic: If the loop completes without finding 7 consecutive identical players, prints "NO".
  std::cout << "NO\n";
}
