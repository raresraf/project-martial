/**
 * @file 266A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 266A - "Stones on the Table").
 *
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color. This is equivalent to
 * counting the number of pairs of adjacent stones that already have the same color.
 *
 * Algorithm: Linear scan and adjacent character comparison.
 * It iterates through the string of stone colors, comparing each stone with
 * its immediate predecessor. A counter is incremented every time two adjacent
 * stones are found to have the same color.
 *
 * Time Complexity: O(N), where N is the length of the string representing the stones.
 * The string is traversed once.
 * Space Complexity: O(1) (excluding input string storage), as only a few variables are used.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads the number of stones and their colors, then calculates and prints
 * the minimum number of stones to remove to satisfy the condition.
 */
int main() {
  int n; // Variable to store the number of stones.
  // Pre-condition: Read the number of stones.
  cin >> n;
  string s; // Variable to store the string of stone colors.
  // Pre-condition: Read the string of stone colors.
  cin >> s;
  int cnt = 0; // Counter for adjacent stones of the same color.
  char a = s[0]; // Stores the color of the previously processed stone, initialized with the first stone.
  // Block Logic: Iterate through the string starting from the second stone.
  // Invariant: 'i' represents the current index in the string.
  for (int i = 1; i < n; i++) {
    // Block Logic: Check if the current stone has the same color as the previous one.
    if (a == s[i]) {
      cnt++; // Increment counter if colors are the same.
    }
    a = s[i]; // Functional Utility: Update 'a' to the current stone's color for the next iteration.
  }
  // Output the total count of stones to remove.
  cout << cnt << endl; // Use endl for flushing output.
  return 0; // Indicate successful execution.
}