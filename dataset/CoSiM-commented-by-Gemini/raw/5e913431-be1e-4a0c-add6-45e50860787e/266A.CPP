/**
 * @file 266A.CPP
 * @brief Solution for the "Stones on the Table" problem from Codeforces.
 * @details The problem requires finding the minimum number of stones to remove from a
 * row of colored stones so that no two adjacent stones have the same color. This is
 * equivalent to counting the number of adjacent identical stones.
 *
 * @algorithm Iterator-based In-place Removal
 * The solution iterates through the string of stones using two iterators, `previous`
 * and `next`, which point to adjacent characters.
 * 1. Initialize `previous` to the first character and `next` to the second.
 * 2. Loop as long as `next` has not reached the end of the string.
 * 3. If the characters at `previous` and `next` are the same, erase the character
 *    at the `next` position. The `std::string::erase` method returns a new valid
 *    iterator to the element that follows the erased one, which is assigned back
 *    to `next`. The `previous` iterator remains unchanged to check against the new `next`.
 *    A counter for removed stones is incremented.
 * 4. If the characters are different, advance both iterators to check the next pair.
 * 5. The final value of the counter is the answer.
 *
 * @complexity
 * - Time: O(N^2) in the worst case, where N is the number of stones. While the main
 *   loop iterates through the string, each call to `std::string::erase` can take
 *   O(N) time as it may need to shift all subsequent characters.
 * - Space: O(N) to store the input string.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: the number of stones (unused in this specific implementation after reading).
  int n;
  // numberOfRemoves: counts the minimum stones to be removed.
  int numberOfRemoves = 0;
  // row: the string representing the colors of the stones.
  string row;

  cin >> n;
  // Consume the newline character left in the input buffer after reading `n`.
  cin.get(); 
  // Read the entire line of stone colors.
  getline(cin, row);

  // `previous` points to the current stone being considered.
  string::iterator previous = row.begin();
  // `next` points to the stone immediately after `previous`.
  string::iterator next = row.begin() + 1;

  /**
   * @brief Traverse the string with two iterators to find and remove adjacent duplicates.
   */
  // Invariant: The substring from the beginning to `previous` has no adjacent duplicates.
  while (next != row.end()) {
    // Pre-condition: Check if the current stone and the next one have the same color.
    if (*previous == *next) {
      // If they are the same, erase the `next` stone.
      // `row.erase()` removes the element and returns an iterator to the
      // element that followed the erased one. This is the correct way to
      // handle iterator invalidation.
      next = row.erase(next);
      numberOfRemoves++;
    } else {
      // If they are different, move both iterators forward to the next pair.
      previous++;
      next++;
    }
  }

  // Print the total number of stones removed.
  cout << numberOfRemoves;

  return 0; // Indicate successful execution.
}
