/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem (Codeforces 266A).
 *
 * This program calculates the minimum number of stones that need to be removed
 * from a row of colored stones such that no two adjacent stones have the same color.
 * The program reads the number of stones and their colors, then counts adjacent
 * identical stones.
 *
 * Algorithm: Linear scan. It iterates through the sequence of stones and increments
 * a counter whenever it finds two adjacent stones of the same color.
 * Time Complexity: O(N) where N is the number of stones.
 * Space Complexity: O(N) to store the stone colors.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Main function to calculate the minimum stones to remove.
 *
 * Reads the number of stones 'n' and their colors. It then counts
 * how many pairs of adjacent stones have the same color, which
 * represents the minimum number of stones to remove.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare 'n' for the number of stones, and 'c' to count adjacent identical stones.
  int n = 0, c = 0;
  // Read the number of stones 'n' from standard input.
  cin >> n;
  // Declare a character array 'ch' of size 'n' to store the stone colors.
  char ch[n];
  /**
   * @brief Block Logic: Reads the colors of 'n' stones.
   *
   * Pre-condition: 'n' specifies the total number of stones.
   * Invariant: 'ch' stores the color of each stone.
   */
  for (int i = 0; i < n; i++)
    cin >> ch[i];

  /**
   * @brief Block Logic: Counts adjacent identical stones.
   *
   * Iterates through the stone colors, comparing each stone with its adjacent
   * one. If they are the same, the counter 'c' is incremented.
   * Pre-condition: 'ch' contains 'n' stone colors.
   * Invariant: 'c' accumulates the count of adjacent identical stone pairs.
   */
  for (int i = 0; i < n; i++)
    // Pre-condition: Check if the current stone and the next stone have the same color.
    if (ch[i] == ch[i + 1])
      c++; // Increment count if colors are identical.

  // Print the total count of adjacent identical stones.
  cout << c;

  return 0;
}
