/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A, "Stones on the Table".
 *
 * Algorithm:
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones share the same color. This implementation
 * uses a `std::list` to store the sequence of colors. It iterates through the
 * list, keeping track of the previous stone's color and incrementing a counter
 * each time a stone's color matches the one before it.
 *
 * Data Structure Choice:
 * The use of `std::list` is unusual for this problem, as a `std::string` or
 * `std::vector` would typically be more direct and potentially more performant
 * due to better cache locality.
 *
 * Time Complexity: O(N), where N is the number of stones. The process involves
 * reading the input (O(N)), copying the string to a list (O(N)), and iterating
 * through the list once (O(N)).
 *
 * Space Complexity: O(N), to store the input string `s` and the `std::list<char> colors`.
 */
#include <cstdio>
#include <iostream>
#include <list>
using namespace std;

int main(int argc, char *argv[]) {
  string s;
  list<char> colors;
  char prev;
  // count: total number of stones.
  // ans: the result, number of stones to remove.
  int count, ans = 0;

  // Read the number of stones and the string representing their colors.
  cin >> count >> s;

  // Edge case: If there's only one stone, no removals are needed.
  if (count == 1) {
    printf("0
");
    return 0;
  }

  /**
   * @brief Copies the colors from the input string to a std::list.
   * This is an intermediate step before processing.
   */
  for (int i = 0; i < count; i++) {
    colors.push_back(s.at(i));
  }

  list<char>::iterator it;
  
  /**
   * @brief Iterates through the list to count adjacent duplicates.
   *
   * Logic:
   * 1. The first color is stored in 'prev'.
   * 2. The first element is removed from the list.
   * 3. The loop then compares each subsequent element to 'prev'.
   * 4. 'prev' is updated in each iteration.
   */
  it = colors.begin();
  prev = *it;
  colors.pop_front();
  for (it = colors.begin(); it != colors.end(); ++it) {
    // If the current color is the same as the previous one, increment the counter.
    if (*it == prev) {
      ans++;
    }
    // Update 'prev' for the next comparison.
    prev = *it;
  }

  // Print the final count of stones to be removed.
  printf("%d
", ans);
  return 0;
}
