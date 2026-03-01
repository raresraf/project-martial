/**
 * @file 266A.CPP
 * @brief Solution for the "Stones on the Table" problem.
 *
 * @problem There are 'n' stones of different colors arranged in a row. The goal is to
 * find the minimum number of stones to remove so that no two adjacent stones
 * have the same color.
 *
 * @algorithm The solution is to iterate through the stones and count the number of
 * adjacent pairs that have the same color. This count is equal to the minimum
 * number of removals required. The algorithm performs a single pass (a single loop)
 * through the string of stone colors.
 *
 * @time_complexity O(N), where N is the number of stones, as it iterates through
 * the string once.
 * @space_complexity O(N) to store the input string of stone colors.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  // `ans` will count the number of stones to be removed.
  int ans = 0;
  string inp;

  // Read the number of stones.
  scanf("%d", &n);

  // The following two lines handle a common issue when mixing scanf and getline/cin.
  // scanf leaves a newline character in the input buffer.
  // getline() consumes that leftover newline, so the subsequent `cin >> inp` works correctly.
  getline(cin, inp);
  cin >> inp;
  
  // Iterate through the stones from the second one to the end.
  for (int i = 1; i < n; i++)
    // If a stone is the same color as the one before it, it needs to be removed.
    if (inp[i] == inp[i - 1])
      ans++;

  // Print the total number of removals required.
  printf("%d", ans);
  return 0;
}
