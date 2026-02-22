/**
 * @file 266A.CPP
 * @brief Solution for the "Stones on the Table" problem (Codeforces 266A).
 * @author Sohanuzzaman Soad
 *
 * @algorithm
 * The problem asks for the minimum number of stones to remove from a row so
 * that no two adjacent stones have the same color.
 *
 * The algorithm is a simple linear scan:
 * 1. Read the number of stones and the string representing their colors.
 * 2. Initialize counters for the number of stones to be removed.
 * 3. Iterate through the stones from the second stone to the end (index 1 to n-1).
 * 4. In each iteration, compare the color of the current stone with the color
 *    of the previous stone.
 * 5. If the colors are the same, it means one stone must be removed. Increment
 *    a counter.
 * 6. After the loop finishes, the total value of the counter(s) is the minimum
 *    number of stones to remove.
 *
 * This specific implementation uses separate counters for each color ('R', 'G', 'B')
 * and sums them at the end, which is a valid, though slightly verbose, approach.
 *
 * @complexity
 * - Time Complexity: O(N), for a single pass through the N stones.
 * - Space Complexity: O(N), to store the input string of colors.
 */
#include <cctype>
#include <iostream>
using namespace std;
/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/
int main() {
  int n;
  cin >> n;
  string a;
  cin >> a;
  // Counters for each color of stone to be removed. A single counter would also suffice.
  int red = 0;
  int green = 0;
  int blue = 0;
  /**
   * @brief Main loop to scan through the stones and count adjacent duplicates.
   * It starts from the second stone (index 1) to compare with the previous one.
   */
  for (int i = 1; i < n; i++) {
    // Check if the current stone has the same color as the one before it.
    if (a[i - 1] == a[i])
      // If they are the same, increment the counter for that specific color.
      if (a[i] == 'R') {
        ++red;
      } else if (a[i] == 'G') {
        ++green;
      } else if (a[i] == 'B') {
        ++blue;
      }
  }
  // The total number of removals is the sum of removals for each color.
  cout << red + green + blue;
  return 0;
}
