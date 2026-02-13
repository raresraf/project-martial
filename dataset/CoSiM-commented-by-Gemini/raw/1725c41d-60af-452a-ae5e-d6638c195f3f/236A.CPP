/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem by counting distinct characters after sorting.
 * @details This program determines if a user is male or female based on whether the number
 * of distinct characters in their username is even or odd. This implementation sorts the
 * string and then iterates through it to count changes in adjacent characters.
 *
 * @section algorithm Algorithm
 * 1. Read the username string `s`.
 * 2. Initialize a `count` of distinct characters to 1. This approach assumes the input string
 *    is non-empty, which is guaranteed by the problem constraints (1 <= length <= 100),
 *    so there is at least one distinct character.
 * 3. Sort the string `s` lexicographically. This places all identical characters next to each other.
 * 4. Iterate through the sorted string from the first character up to the second-to-last character.
 * 5. In each iteration, compare the current character `s[i]` with the next one `s[i+1]`.
 * 6. If the characters are different, it signifies a change to a new distinct character, so `count` is incremented.
 * 7. After the loop, `count` holds the total number of distinct characters.
 * 8. Check the parity of `count` and print "CHAT WITH HER!" for even, or "IGNORE HIM!" for odd.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L log L), where L is the length of the string, dominated by the `std::sort` operation.
 * - Space Complexity: O(L) to store the string; the sorting and counting are done with O(1) auxiliary space.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;
  // Initialize count to 1, assuming the input string is non-empty.
  int count = 1;

  // Sort the string to group identical characters together.
  sort(s.begin(), s.end());

  /**
   * @brief Counts distinct characters by detecting changes in the sorted string.
   * This loop increments the count each time a character differs from the one
   * immediately following it.
   */
  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] != s[i + 1])
      count++;
  }
  // cout << count << endl;
  // The final decision is based on the parity of the distinct character count.
  if (count % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
