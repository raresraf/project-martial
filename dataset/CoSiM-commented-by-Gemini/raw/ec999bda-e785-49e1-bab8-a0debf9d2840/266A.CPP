/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid adjacent same-colored stones.
 * 
 * The problem is to determine the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. This is equivalent to
 * counting the number of pairs of adjacent stones that are of the same color.
 * 
 * @b Algorithm: Linear scan. It iterates through the sequence of stones and counts
 *    every instance where a stone is the same color as the one preceding it.
 * @b Time_Complexity: O(N), where N is the number of stones, due to the single pass
 *    required for both input reading and counting.
 * @b Space_Complexity: O(N) to store the sequence of stones.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;      // Number of stones.
  int ct = 0; // Counter for stones to be removed.
  vector<char> s;

  // Read the number of stones.
  cin >> n;
  
  // Read the sequence of stone colors.
  for (int i = 0; i < n; i++) {
    char c;
    cin >> c;
    s.push_back(c);
  }

  /**
   * @brief Iterates through the stones to find adjacent pairs of the same color.
   * @invariant `ct` holds the count of stones to remove from the prefix of the
   *    sequence that has been processed.
   */
  for (int i = 1; i < n; i++) {
    // If a stone has the same color as its immediate neighbor to the left,
    // one of them must be removed.
    if (s[i] == s[i - 1])
      ct++;
  }

  // Output the total count of stones to remove.
  cout << ct << endl;
}