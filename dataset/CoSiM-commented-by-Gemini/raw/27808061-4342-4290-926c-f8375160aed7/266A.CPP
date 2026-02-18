/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem (266A) from Codeforces.
 * @details The problem asks for the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution iterates through the string of stones and removes one of
 * any pair of adjacent, identical stones.
 * 1. Read the number of stones (n) and the string representing their colors.
 * 2. Initialize a counter for removed stones to 0.
 * 3. Iterate through the string from the second stone onwards.
 * 4. At each position `i`, compare the stone with the one at `i-1`.
 * 5. If they are the same color:
 *    a. Erase the stone at `i-1` from the string.
 *    b. Decrement the loop counter `i` to account for the shortened string, ensuring the
 *       next comparison is with the newly adjacent stone.
 *    c. Increment the removal counter.
 * 6. After the loop, print the total count of removed stones.
 *
 * @complexity
 * Time Complexity: O(N^2), where N is the number of stones. The loop runs up to N times,
 * and the `string::erase` operation on a std::string can take O(N) time in the worst case
 * as it may require shifting subsequent elements.
 * Space Complexity: O(N) to store the string of stone colors.
 */
#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;

  string str;
  cin >> str;
  int count = 0;

  /**
   * @block
   * @brief Iterates through the stones, removing adjacent duplicates.
   * @invariant 'count' holds the number of stones removed so far.
   * @pre 'str' contains the initial sequence of stone colors.
   * @post 'str' is modified to have no adjacent duplicates, and 'count' is the total number removed.
   */
  for (int i = 1; i < str.size(); i++) {
    // Check if the current stone has the same color as the previous one.
    if (str[i] == str[i - 1]) {
      // If they are the same, remove the previous stone.
      str.erase(str.begin() + i - 1);
      // Decrement 'i' to re-check the current position, as the string has shifted.
      i--;
      count++;
    }
  }

  cout << count;
  return 0;
}
