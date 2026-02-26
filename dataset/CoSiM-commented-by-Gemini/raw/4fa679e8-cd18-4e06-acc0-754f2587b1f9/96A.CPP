/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem by manually counting consecutive characters.
 *
 * @details The problem is to determine if a situation in a football match is
 * "dangerous", which occurs if there are at least seven consecutive players
 * from the same team ('0' or '1').
 *
 * @algorithm
 * This solution iterates through the string character by character, keeping a
 * running count of the current sequence of identical characters.
 * 1. Initialize a counter to 1 and a variable to hold the previous character.
 * 2. Loop through each character of the input string.
 * 3. If the current character is the same as the previous one, increment the counter.
 * 4. If it's different, reset the counter to 1.
 * 5. After each check, if the counter has reached 7, a dangerous situation is
 *    found, and the loop can be terminated early.
 * 6. After the loop, check the final value of the counter to determine the output.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string, due to the single
 * linear pass.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string> // It's good practice to include the headers you use.

int main() {
  std::string s;
  std::cin >> s;

  // `count` tracks the length of the current run of identical characters.
  auto count = 1;
  // `temp` stores the character from the previous iteration for comparison.
  auto temp = ' ';

  /**
   * @block
   * @brief Main loop to find a run of 7 identical characters.
   */
  for (auto letter : s) {
    if (temp == letter) {
      // The run of identical characters continues.
      count++;
    } else {
      // A new character is found, so reset the run counter.
      count = 1;
    }
    // Update the previous character for the next iteration.
    temp = letter;

    // If a run of 7 is found, we can stop early.
    if (count == 7) {
      break;
    }
  }

  /**
   * @block
   * @brief Print the result based on the final count.
   */
  if (count == 7) {
    std::cout << "YES" << std::endl;
  } else {
    std::cout << "NO" << std::endl;
  }

  return 0;
}