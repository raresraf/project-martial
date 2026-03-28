/**
 * @file 58A.CPP
 * @brief Determines if the word "hello" is a subsequence of a given input string.
 *
 * This program checks if the characters of "hello" appear in the input string
 * in the correct order, without necessarily being contiguous.
 *
 * @algorithm The code uses an index-based approach. It first collects the indices of all
 * occurrences of the characters 'h', 'e', 'l', and 'o'. It then attempts to find a
 * valid set of indices (i_h, i_e, i_l1, i_l2, i_o) from the collected lists such
 * that i_h < i_e < i_l1 < i_l2 < i_o.
 *
 * @time_complexity O(N), where N is the length of the input string, due to the initial pass.
 * @space_complexity O(N) in the worst case, as the index vectors can grow up to the size of the input string.
 */
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
// Vectors to store the 0-based indices of each required character from the input string.
vector<int> h, e, l, o;
// Array to store the selected indices for the characters of "hello".
// cnt[0]=h, cnt[1]=e, cnt[2]=l, cnt[3]=l, cnt[4]=o
int cnt[5];
int main() {
  string s;
  cin >> s;
  /**
   * @block_logic This loop populates the vectors with the indices of each
   * character 'h', 'e', 'l', 'o' found in the input string `s`.
   */
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == 'h')
      h.push_back(i);
    if (s[i] == 'e')
      e.push_back(i);
    if (s[i] == 'l')
      l.push_back(i);
    if (s[i] == 'o')
      o.push_back(i);
  }

  /**
   * @block_logic This section attempts to find a valid, ordered set of indices for "hello".
   * It works by anchoring the search with the first 'h' and last 'o' and then
   * finding the intermediate characters ('e', 'l', 'l') that fit between these anchors
   * and in the correct relative order.
   */
  // Pre-condition: Check if 'h' and 'o' exist in the string at all.
  if (h.size() != 0 && o.size() != 0) {
    // Anchor: Select the first 'h' and the last 'o'.
    cnt[0] = h[0];
    cnt[4] = o[o.size() - 1];
  }
  // Find the first 'e' that appears after the selected 'h'.
  for (int i = 0; i < e.size(); ++i)
    if (e[i] > cnt[0]) {
      cnt[1] = e[i];
      break;
    }
  // Find the last 'l' that appears before the selected 'o' to act as the second 'l'.
  for (int i = l.size() - 1; i >= 0; --i)
    if (l[i] < cnt[4]) {
      cnt[3] = l[i];
      break;
    }
  // Find the first 'l' that is between the selected 'e' and the second 'l'.
  for (int i = 0; i < l.size(); ++i)
    if (l[i] < cnt[3] && l[i] > cnt[1]) {
      cnt[2] = l[i];
      break;
    }

  /**
   * @block_logic Validation check. This confirms that a valid set of indices was found
   * and that they are in strictly increasing order, which confirms "hello" as a subsequence.
   * Note: If any character was not found, its index in `cnt` would be 0, likely failing the check.
   */
  if (cnt[0] < cnt[1] && cnt[1] < cnt[2] && cnt[2] < cnt[3] &&
      cnt[3] < cnt[4]) {
    cout << "YES";
    return 0;
  }
  cout << "NO";
  return 0;
}
