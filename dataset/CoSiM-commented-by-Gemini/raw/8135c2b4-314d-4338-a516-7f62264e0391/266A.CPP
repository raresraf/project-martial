/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid adjacent same-colored stones.
 *
 * This program solves the "Stones on the Table" problem. Given a row of stones
 * of different colors (R, G, B), it calculates the minimum number of stones
 * that need to be removed so that no two adjacent stones have the same color.
 *
 * @algorithm Greedy Approach
 * @complexity
 * Time: O(N), where N is the number of stones, as we iterate through the
 *       row of stones exactly once.
 * Space: O(N) to store the input string representing the stone colors.
 */
#include <cstdio>
#include <iostream>
#include <string> // Added for std::string
using namespace std;

/**
 * @brief Main function to handle input and calculate the result.
 */
int main() {
  int length;
  // The "
" in scanf is crucial to consume the newline character after the
  // integer, so that getline works correctly on the next line.
  scanf("%d
", &length);
  string input;
  getline(cin, input);

  // total: counts the number of stones to be removed.
  // index: loop counter for iterating through the stones.
  int total = 0, index = 0;
  // state: holds the color of the last stone that was not removed.
  char state = input[0];

  // Pre-condition: 'state' is the color of the first stone. 'total' is 0.
  // Block Logic: Iterate through the stones starting from the second one.
  while (++index < length) {
    // Compare the current stone's color with the color of the last valid stone.
    if (input[index] == state) {
      // If the colors are the same, this stone must be removed.
      ++total;
    } else {
      // If the colors are different, this stone becomes the new reference state.
      state = input[index];
    }
  }
  
  // Print the total count of stones to be removed.
  printf("%d
", total);
  return 0;
}
