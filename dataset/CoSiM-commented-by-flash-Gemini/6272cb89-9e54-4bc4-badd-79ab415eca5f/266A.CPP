/**
 * @6272cb89-9e54-4bc4-badd-79ab415eca5f/266A.CPP
 * @brief This program solves problem 266A, "Stones on the Table".
 * Given a string `inp` representing a sequence of colored stones, the objective
 * is to find the minimum number of stones to remove such that no two adjacent
 * stones have the same color. This is achieved by counting adjacent duplicate stones.
 *
 * Algorithm: Single-pass linear scan to count adjacent identical characters.
 * Time Complexity: O(N) where N is the length of the input string.
 * Space Complexity: O(N) for storing the input string.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to execute the Stones on the Table problem logic.
 * Reads the number of stones and the string representing their colors,
 * then calculates and prints the minimum number of stones to remove.
 * @return 0 on successful execution.
 */
int main() {
  int n, ans = 0; // Declares 'n' for the number of stones and 'ans' (count of removals) initialized to 0.
  string inp;     // Declares a string variable to store the input stone sequence.
  
  // Block Logic: Reads the integer 'n' (number of stones).
  // `scanf` reads the integer but leaves the newline character in the buffer.
  scanf("%d", &n); 
  
  // Block Logic: Consumes the remaining newline character from the input buffer.
  // This is necessary because `cin >> inp` (or `getline`) would otherwise read the leftover newline.
  getline(cin, inp); 
  
  // Block Logic: Reads the string representing the sequence of stones.
  // Since `getline` might have consumed the newline and `cin >> inp` reads until whitespace,
  // this is usually intended to read the actual string after `n`.
  cin >> inp; 
  
  /**
   * Block Logic: Iterates through the stone sequence to count adjacent identical stones.
   * Invariant: `ans` accumulates the total number of stones that need to be removed.
   */
  for (int i = 1; i < n; i++) // Loop starts from the second stone (index 1).
    // Inline: Checks if the current stone is the same color as the previous stone.
    if (inp[i] == inp[i - 1]) 
      ans++; // Increments 'ans' if adjacent stones have the same color.
  
  // Block Logic: Prints the final count of stones to be removed.
  printf("%d", ans); 
  return 0;
}