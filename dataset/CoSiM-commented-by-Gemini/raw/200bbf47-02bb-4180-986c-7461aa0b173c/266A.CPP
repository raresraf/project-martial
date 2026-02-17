/**
 * @file 266A.CPP
 * @brief Solution for the "Stones on the Table" competitive programming problem.
 *
 * @details This program calculates the minimum number of stones to remove from a
 * row so that no two adjacent stones have the same color. This is equivalent to
 * counting the number of adjacent, identical stones.
 *
 * @section algorithm
 * Algorithm:
 * A greedy linear scan is performed on the string of stones.
 * 1. Read the number of stones, `n`, and the string representing their colors.
 * 2. Initialize a removal counter to zero.
 * 3. Iterate through the string, comparing each stone's color with the color of
 *    the stone next to it.
 * 4. If two adjacent stones have the same color, increment the counter.
 * 5. Print the total count, which represents the minimum number of removals.
 *
 * @section complexity
 * Time Complexity: O(N), where N is the number of stones, due to the single
 * pass through the input string.
 * Space Complexity: O(N), to store the string of stone colors.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, n, c = 0;
  cin >> n;
  string s;
  cin >> s;

  // Block Logic: Iterate through the string to count adjacent identical stones.
  // The loop uses a C-style null terminator check (`s[i] != '\0'`). This works
  // for std::string as s[n] is guaranteed to be '\0'. The loop effectively
  // checks pairs (s[0], s[1]), (s[1], s[2]), ..., (s[n-1], s[n]).
  for (i = 0; s[i] != '\0'; i++) {
    if (s[i] == s[i + 1])
      c++;
  }

  // Output the total number of stones to remove.
  cout << c;
}
