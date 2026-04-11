/**
 * @file 266A.CPP
 * @brief Solution to a competitive programming problem: "Stones on the Table".
 * This program reads an integer 'n' (representing the number of stones) and then
 * a sequence of 'n' characters (representing the colors of stones). It counts
 * the number of adjacent stones that have the same color.
 * Algorithm: Iterative character comparison with a counter.
 * Time Complexity: O(N) where N is the number of stones.
 * Space Complexity: O(1)
 */
#include <cctype>   // Potentially for character handling, though not directly used in the logic.
#include <iostream> // For input/output operations (cin, cout).
#include <string>   // For string manipulation, though characters are read one by one.

using namespace std;

/**
 * @brief Main function where the program execution begins.
 * Reads stone colors and counts adjacent identical colors.
 */
int main() {
  // freopen("input.txt", "r", stdin); // Commented out: typically used for redirecting input from a file.
  int n; // Declares an integer 'n' to store the number of stones.
  // Reads the number of stones.
  cin >> n;
  char x, y;    // 'x' stores the previous stone's color, 'y' stores the current stone's color.
  int cnt = 0;  // Counter for adjacent identical stones.
  // Reads the color of the first stone.
  cin >> x;
  /**
   * Block Logic: Iterates through the remaining stones, comparing each with the previous one.
   * Invariant: 'x' always holds the character of the last non-duplicate stone encountered.
   * Pre-condition: 'n' stones follow the initial stone 'x'.
   */
  while (cin >> y) { // Reads stone colors one by one until end of input.
    /**
     * Block Logic: Checks if the current stone 'y' has a different color than the previous 'x'.
     * Pre-condition: 'x' and 'y' hold valid character representations of stone colors.
     */
    if (y != x)
      // If colors are different, updates 'x' to the current stone's color.
      x = y;
    else
      // If colors are the same, increments the counter.
      cnt++;
  }
  // Prints the total count of adjacent identical stones.
  cout << cnt;
  return 0;
}
