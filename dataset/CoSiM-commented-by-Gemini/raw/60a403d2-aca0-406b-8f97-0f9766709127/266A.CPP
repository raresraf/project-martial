/**
 * @file 266A.CPP
 * @brief Solution for the "Stones on the Table" problem.
 *
 * @problem There are 'n' stones of colors Red, Green, or Blue, arranged in a row.
 * The goal is to find the minimum number of stones to remove so that no two
 * adjacent stones have the same color.
 *
 * @algorithm The solution is to iterate through the stones and count the number of
 * adjacent pairs that have the same color. This count directly corresponds to the
 * minimum number of removals required. The algorithm performs a single pass
 * through the string of stone colors.
 *
 * @time_complexity O(N), where N is the number of stones, because we iterate
 * through the string once.
 * @space_complexity O(N) to store the input string representing the stones.
 */
#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  // This counter will track the number of stones to be removed.
  int count = 0;
  cin >> n;
  string str;
  cin >> str;

  // Iterate through the stones from the second stone to the end.
  // Pre-condition: The loop starts at index 1 to allow comparison with the previous element.
  for (int i = 1; i < n; i++) {
    // Check if the current stone has the same color as the one before it.
    if (str[i] == str[i - 1]) {
      // If they are the same, one must be removed. Increment the counter.
      count++;
    }
  }

  // Output the total number of stones that need to be removed.
  std::cout << count;
  return 0;
}
