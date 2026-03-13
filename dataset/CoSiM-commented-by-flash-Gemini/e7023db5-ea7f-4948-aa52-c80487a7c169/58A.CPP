/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem (Codeforces 58A) by checking if "hello"
 * is a subsequence of an input string.
 *
 * @section Algorithm
 * This implementation uses a non-standard, index-gathering approach. Instead of a
 * simple greedy scan, it first collects the indices of all occurrences of the
 * required characters ('h', 'e', 'l', 'o') into separate vectors. It then
 * attempts to find a valid, strictly increasing sequence of indices corresponding
 * to the letters of "hello".
 *
 * The logic is as follows:
 * 1. Pre-processing: Iterate through the input string once to populate four vectors
 *    with the indices of all 'h's, 'e's, 'l's, and 'o's.
 * 2. Heuristic Search: The code tries to find a valid set of indices by making
 *    several assumptions and searches:
 *    - It picks the index of the *first* 'h'.
 *    - It picks the index of the *last* 'o'.
 *    - It finds the *first* 'e' that appears after the chosen 'h'.
 *    - It finds the *last* 'l' that appears before the chosen 'o' (for the second 'l' in "hello").
 *    - It finds another 'l' that appears between the chosen 'e' and the second 'l'.
 * 3. Validation: Finally, it checks if the five chosen indices are in a strictly
 *    increasing order. If they are, "hello" is a subsequence.
 *
 * This method is more complex and less robust than the standard two-pointer greedy algorithm.
 *
 * @section Complexity
 * - Time Complexity: O(L), where L is the length of the input string. The initial
 *   pass and the subsequent searches through index vectors all take linear time in total.
 * - Space Complexity: O(L) in the worst case, to store the indices of all characters.
 */
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
// Vectors to store the indices of each character's occurrences.
vector<int> h, e, l, o;
// Array to store the chosen indices for the 'h', 'e', 'l', 'l', 'o' sequence.
int cnt[5];
int main() {
  string s;
  cin >> s;

  /**
   * @brief Block Logic: Pre-processing step to gather all indices.
   * This loop iterates through the input string and populates the respective
   * vectors with the indices where 'h', 'e', 'l', and 'o' appear.
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

  // Initial check to ensure the first and last characters exist.
  if (h.size() != 0 && o.size() != 0) {
    // Greedily select the first 'h' and the last 'o' as boundary candidates.
    cnt[0] = h[0];
    cnt[4] = o[o.size() - 1];
  }

  /**
   * @brief Block Logic: Find the first 'e' that appears after the first 'h'.
   */
  for (int i = 0; i < e.size(); ++i)
    if (e[i] > cnt[0]) {
      cnt[1] = e[i];
      break;
    }
  
  /**
   * @brief Block Logic: Find the last 'l' that appears before the last 'o'. This
   * will serve as the second 'l' in "hello".
   */
  for (int i = l.size() - 1; i >= 0; --i)
    if (l[i] < cnt[4]) {
      cnt[3] = l[i];
      break;
    }
  
  /**
   * @brief Block Logic: Find the first 'l' that appears after the 'e' and before
   * the second 'l'. This will serve as the first 'l' in "hello".
   */
  for (int i = 0; i < l.size(); ++i)
    if (l[i] < cnt[3] && l[i] > cnt[1]) {
      cnt[2] = l[i];
      break;
    }

  /**
   * @brief Final Validation: Check if the collected indices form a strictly
   * increasing sequence, which would confirm "hello" as a subsequence.
   */
  if (cnt[0] < cnt[1] && cnt[1] < cnt[2] && cnt[2] < cnt[3] &&
      cnt[3] < cnt[4]) {
    cout << "YES";
    return 0;
  }

  // If no valid increasing sequence of indices was found.
  cout << "NO";
  return 0;
}