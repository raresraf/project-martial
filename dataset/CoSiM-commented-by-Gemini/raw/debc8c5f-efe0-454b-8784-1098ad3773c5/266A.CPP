#include <iostream>
#include <string>
using namespace std;

/**
 * @file 266A.CPP
 * @brief Solution for a simple string-based counting problem, "Stones on the
 * Table".
 *
 * Problem Statement:
 * There are `n` stones on a table in a row, each with a color ('R', 'G', or
 * 'B'). The goal is to calculate the minimum number of stones to remove so
 * that no two adjacent stones have the same color.
 *
 * Algorithm:
 * The problem can be solved by iterating through the stones and counting how
 * many stones need to be removed. A stone needs to be removed if it has the
 * same color as the stone immediately preceding it.
 *
 * The implemented logic iterates from the second stone and compares each stone
 * to the last "kept" stone.
 * 1. Read the number of stones `n` and the string of colors `word`.
 * 2. Initialize a counter `cont` for removed stones to 0.
 * 3. Initialize a position marker `pos` to 0, representing the index of the
 *    last stone that is part of the valid (no adjacent same-colors) sequence.
 * 4. Loop from the second stone (`i = 1`) to the end.
 * 5. If the current stone `word[i]` has the same color as the stone at `pos`,
 *    it's a stone to be removed, so increment `cont`.
 * 6. If the colors are different, the current stone becomes the new "last kept"
 *    stone, so update `pos` to `i`.
 * 7. The final value of `cont` is the answer.
 *
 * @note A slightly simpler implementation would be to just compare `word[i]`
 * with `word[i-1]` in each iteration of the loop.
 *
 * Time Complexity: O(N), where N is the number of stones.
 * Space Complexity: O(N) to store the input string.
 */
int main() {
  int n;
  int pos = 0;  // The index of the last stone that wasn't a duplicate.
  int cont = 0; // Counter for the number of stones to remove.
  string word;

  cin >> n;
  cin >> word;

  // Block Logic: Iterate through the stones starting from the second one.
  // Invariant: 'cont' is the number of stones removed from the first 'i' stones.
  // 'pos' is the index of the last valid stone in the sequence up to i-1.
  for (int i = 1; i < n; i++) {
    // Pre-condition: Check if the current stone has the same color as the
    // last valid stone in the sequence.
    if (word[i] == word[pos]) {
      // If so, this stone is a consecutive duplicate and must be removed.
      cont++;
    } else {
      // Otherwise, this stone is not a duplicate; it becomes the new
      // reference point for the next comparisons.
      pos = i;
    }
  }

  cout << cont << endl;
  return 0;
}