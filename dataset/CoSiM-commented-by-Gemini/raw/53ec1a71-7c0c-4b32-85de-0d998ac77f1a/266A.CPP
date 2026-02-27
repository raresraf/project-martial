/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces.
 * @details The problem requires finding the minimum number of stones to take
 * from a row so that no two adjacent stones have the same color.
 *
 * @algorithm
 * This solution uses a stack to find the number of necessary removals. The stack
 * represents the final, valid arrangement of stones where no two adjacent stones
 * are the same color.
 * 1. Read the number of stones and the string of colors.
 * 2. Initialize an empty stack of characters and a removal counter `cnt` to 0.
 * 3. Iterate through each stone (character) in the input string.
 * 4. For each stone:
 *    a. Check if the stack is empty or if the current stone's color is
 *       different from the color of the stone at the top of the stack.
 *    b. If it is different (or the stack is empty), push the current stone
 *       onto the stack. This "keeps" the stone.
 *    c. If the current stone's color is the same as the one at the top of the
 *       stack, it's a consecutive duplicate that must be removed. Increment
 *       the counter `cnt` and do not push the stone onto the stack.
 * 5. After the loop, `cnt` holds the total number of removed stones.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of stones, due to a single pass.
 * Space Complexity: O(n) in the worst case for the stack (e.g., "RGBRGB...").
 */
#include <bits/stdc++.h>
#include <iostream>
#define ll long long
#define MOD 1000000007
using namespace std;

int main() {
  // Fast I/O
  ios::sync_with_stdio(false);
  int n;
  cin >> n;
  string s;
  cin >> s;
  
  stack<char> st; // The stack holds the valid, non-repeating sequence of stones.
  int cnt = 0;    // Counter for the number of stones to be removed.

  /**
   * @block
   * @brief Iterate through each stone in the row.
   */
  for (char ch : s) {
    // Check if the current stone is different from the last valid stone.
    if (st.empty() || st.top() != ch)
      // If it's the first stone or a different color, keep it.
      st.push(ch);
    else
      // If it's the same as the last stone, it's a duplicate to be removed.
      cnt++;
  }

  cout << cnt << endl;
  return 0;
}
