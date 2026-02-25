/**
 * @file 266A.CPP
 * @brief Solution for a programming problem, "Stones on the Table".
 * @details The problem asks for the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color.
 *
 * Algorithm:
 * The problem can be solved by a single pass through the row of stones.
 * 1. Read the number of stones, 'n', and the sequence of their colors.
 * 2. Initialize a counter for removals to 0.
 * 3. Iterate from the first stone to the second-to-last stone.
 * 4. In each step, compare the color of the current stone with the color of the next one.
 * 5. If they are the same, it means one of them must be removed to satisfy the
 *    condition. Increment the removal counter.
 * 6. The final value of the counter is the minimum number of stones to remove.
 *
 * Note on Implementation: The code uses a VLA `char ch[n]`, which is a non-standard
 * C++ extension. The loop `for (int i = 0; i < n; i++)` has a potential out-of-bounds
 * access when `i = n - 1` as it checks `ch[i + 1]`.
 *
 * Time Complexity: O(n), where 'n' is the number of stones.
 * Space Complexity: O(n) to store the sequence of colors.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n = 0, c = 0;
  cin >> n;
  char ch[n]; // Using a Variable Length Array (VLA), a non-standard C++ extension.
  
  // Read the sequence of stone colors.
  for (int i = 0; i < n; i++)
    cin >> ch[i];

  /**
   * @block
   * @description This loop iterates through the stones to count adjacent pairs
   * of the same color. Each such pair requires one stone to be removed.
   */
  for (int i = 0; i < n; i++)
    // Check if the current stone and the next one have the same color.
    if (ch[i] == ch[i + 1])
      c++;

  // Output the total number of stones to be removed.
  cout << c;

  return 0;
}