/**
 * @file 266A.CPP
 * @brief This program solves the "Stones on the Table" problem.
 * It calculates the minimum number of stones to remove from a row such that no two adjacent
 * stones have the same color.
 * Algorithm: Linear scan through the stone colors, counting adjacent identical colors.
 * Time Complexity: O(N), where N is the total number of stones, due to a single pass.
 * Space Complexity: O(N) for storing the stones in a list.
 */
#include <cstdio>   // Required for printf.
#include <iostream> // Required for cin.
#include <list>     // Required for std::list.
using namespace std;

int main(int argc, char *argv[]) {
  string s;            // Stores the input string of stone colors.
  list<char> colors;   // A list to hold individual stone colors for easier manipulation.
  char prev;           // Stores the color of the previous stone for comparison.
  int count;           // Stores the total number of stones.
  int ans = 0;         // Stores the count of stones to be removed.

  // Reads the total number of stones ('count') and the string of colors ('s').
  cin >> count >> s;

  /**
   * Conditional Logic: Handles the base case where there is only one stone.
   * If there's only one stone, no removals are needed, so it prints 0 and exits.
   */
  if (count == 1) {
    printf("0\n");
    return 0;
  }

  /**
   * Block Logic: Populates the `colors` list with characters from the input string `s`.
   * This prepares the list for sequential processing.
   * Invariant: The `colors` list accurately represents the sequence of stone colors.
   */
  for (int i = 0; i < count; i++) {
    colors.push_back(s.at(i));
  }

  // Declares an iterator for the list.
  list<char>::iterator it;
  // Initializes the iterator to the beginning of the list.
  it = colors.begin();
  // Stores the color of the first stone as the 'previous' stone.
  prev = *it;
  // Removes the first stone from the list, as it has been accounted for as 'prev'.
  colors.pop_front();
  /**
   * Block Logic: Iterates through the remaining stones in the list to count adjacent stones of the same color.
   * 'ans' is incremented each time a stone has the same color as its predecessor.
   * Invariant: 'ans' correctly accumulates the number of stones that need to be removed to satisfy the condition.
   */
  for (it = colors.begin(); it != colors.end(); ++it) {
    // If the current stone's color is the same as the previous stone's color.
    if (*it == prev) {
      ans++; // Increment the count of stones to remove.
    }
    prev = *it; // Update 'prev' to the current stone's color for the next comparison.
  }

  // Prints the total number of stones that need to be removed.
  printf("%d\n", ans);
  // Standard program exit.
  return 0;
}
