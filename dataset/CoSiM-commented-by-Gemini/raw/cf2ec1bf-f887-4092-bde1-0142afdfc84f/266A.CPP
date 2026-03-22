/**
 * @file 266A.CPP
 * @brief Calculates the minimum number of stones to remove to avoid adjacent identical colors.
 *
 * The problem consists of a row of stones, each with a specific color (R, G, or B).
 * The goal is to find the minimum number of stones to remove so that no two adjacent
 * stones have the same color. This implementation simulates the removal process directly.
 *
 * @b Algorithm: Direct simulation with in-place vector removal.
 * @b Time_Complexity: O(N^2) in the worst case, due to the linear-time complexity of
 *   `vector::erase` within a loop.
 * @b Space_Complexity: O(N) to store the sequence of stones in a vector.
 *
 * @note The implementation has a logic flaw. When an element is erased from the vector,
 * the iterator is unconditionally incremented in the for-loop header. This causes it to
 * skip checking the element that shifts into the next position, leading to incorrect
 * results for inputs with three or more consecutive identical characters (e.g., "RRR").
 */
// http://codeforces.com/problemset/problem/266/A
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string input;
  cin >> n;
  cin >> input;
  
  // A vector to hold the characters representing the stones.
  vector<char> stones;
  // Pre-condition: The input string is copied to a vector of characters.
  for (int i = 0; i < n; i++)
    stones.push_back(input[i]);

  // 'removals' counts the number of stones removed.
  int removals = 0;

  /**
   * @brief Iterates through the stones to find and remove adjacent duplicates.
   * This loop contains a logic error. It incorrectly advances the iterator after
   * removing an element, causing it to miss consecutive duplicates beyond a pair.
   */
  for (auto itr = stones.begin(); itr < stones.end() - 1; itr++) {
    // If an adjacent pair of stones has the same color...
    if (*itr == *(itr + 1)) {
      // ...remove the second stone of the pair and increment the counter.
      stones.erase(itr + 1);
      removals++;
    }
  }
  
  // Output the total number of removals.
  cout << removals << endl;
  return 0;
}
