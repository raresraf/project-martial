/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem (Codeforces 266A).
 *
 * Problem Statement:
 * There are 'n' stones arranged in a row, with colors represented by a string
 * (e.g., 'R', 'G', 'B'). The objective is to calculate the minimum number of
 * stones that need to be removed so that no two adjacent stones have the same color.
 *
 * Algorithm:
 * The program employs a simple and direct linear scan approach. It iterates
 * through the string of stones from the first to the second-to-last stone. In
 * each step, it compares the color of the current stone `s[i]` with the color
 * of the next stone `s[i+1]`. If they are the same, it means one of these two
 * stones must be removed to satisfy the condition. A counter is incremented for
 * every such adjacent pair found. This single pass correctly counts the total
 * number of stones to be removed.
 *
 * Time Complexity: O(n), where 'n' is the number of stones, because it involves
 * a single loop through the input string.
 * Space Complexity: O(n) to store the input string 's'.
 */
// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;

  // Handle the edge case of zero stones.
  if (!n) {
    cout << 0 << endl;
    return 0;
  }

  int count = 0; // This will count the minimum number of stones to remove.
  /**
   * Block Logic: Iterate through the stones to find adjacent pairs of the same color.
   * The loop runs up to n-1 to prevent out-of-bounds access on s[i+1].
   */
  for (int i = 0; i < n - 1; i++) {
    // If a stone has the same color as its immediate neighbor, one must be removed.
    if (s[i] == s[i + 1])
      count++;
  }

  // Print the total number of removals required.
  cout << count << endl;
  return 0;
}
