/**
 * @file 236A.CPP
 * @author Andr s Sierra
 * @brief Solution to a competitive programming problem, likely Codeforces 236A "Boy or Girl".
 *
 * @details
 * The problem is to determine a user's gender based on their username. If the
 * number of distinct characters in the username is odd, the user is considered
 * male ("IGNORE HIM!"). If the number of distinct characters is even, the user
 * is considered female ("CHAT WITH HER!").
 *
 * @algorithm
 * The implemented algorithm attempts to count unique characters using a nested
 * loop, which has O(N^2) complexity. It iterates through the string, and for
 * each character, it scans the rest of the string for duplicates.
 *
 * @flaw
 * This implementation contains a major logical flaw. The check to print the
 * final answer is located *inside* the main `while` loop. This causes the
 * program to make a decision and print an answer after processing only a
 * partial, and likely incorrect, count of the unique characters, rather than
 * waiting for the full count to be computed. A standard, correct approach would
 * use a `std::set` or sort the string to count unique characters in O(N log N) time.
 *
 * @complexity
 * - Time Complexity: O(N^2), due to the nested loop structure for finding duplicates.
 * - Space Complexity: O(N), for storing two copies of the input string.
 */

#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {

  string str;

  cin >> str;

  string strTest = str; // A copy of the string for comparison.
  int cnt = str.length(); // Starts by assuming all characters are unique.
  unsigned int i = 0;

  /**
   * @brief This loop attempts to find and discount duplicate characters.
   */
  while (i != str.length()) {
    // For each character `str[i]`, scan the rest of the string for duplicates.
    for (unsigned int j = i + 1; j <= str.length(); j++) {
      if (str[i] == strTest[j]) {
        // If a duplicate is found, decrement the count and stop scanning for this char.
        cnt--;
        break;
      }
    }

    i++;

    /**
     * @brief FLAW: This check is inside the main loop.
     * The program will print an output on every iteration where `i == str.length()`
     * is true. Since the loop condition is `i != str.length()`, this block is
     * only entered once the loop has finished, but its placement is confusing
     * and suggests a misunderstanding of the loop's scope. The count `cnt` at this
     * point is not a correct count of unique characters due to the flawed logic.
     */
    if (i == str.length()) {
      if (cnt % 2 == 0)
        cout << "CHAT WITH HER!" << endl;
      else
        cout << "IGNORE HIM!" << endl;
    }
  }
}
