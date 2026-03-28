/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces.
 *
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones share the same color. This is equivalent to
 * counting the number of adjacent, identical characters in a string.
 *
 * @algorithm The code iterates through the string from the second character to the
 * end. In each step, it compares the current character with its immediate
 * predecessor. If they are identical, a counter is incremented. This counter
 * represents the total number of stones that need to be removed.
 *
 * @time_complexity O(N), where N is the number of stones (the length of the string).
 * @space_complexity O(N) to store the input string.
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n; // Note: n is read but s.size() is used, making n redundant.
  string s;
  cin >> s;
  int ans = 0;
  /**
   * @block_logic This loop iterates from the second character to the end of the
   * string, comparing each character with the one before it.
   */
  for (int i = 1; i < s.size(); ++i)
    // If a character is the same as its predecessor, it's a pair that needs one removal.
    if (s[i] == s[i - 1])
      ++ans;
  cout << ans;
}