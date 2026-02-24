/**
 * @file raw/46659472-aabf-4c6a-af1f-2493e79c2466/236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem is to determine if the number of distinct characters in a username
 * is even or odd. The user is considered female ("CHAT WITH HER!") if the count
 * is even, and male ("IGNORE HIM!") if it's odd.
 *
 * This solution uses a sorting-based approach to count the distinct characters.
 *
 * The algorithm is as follows:
 * 1. Read the username into a `std::string`.
 * 2. Sort the string alphabetically using `std::sort`. This groups all
 *    identical characters together (e.g., "uncopyrightable" -> "aabceghilnoprtuy").
 * 3. Initialize a counter `x` for distinct characters.
 * 4. Iterate through the sorted string. In each step, compare the current
 *    character `s[i]` with the next character `s[i+1]`.
 * 5. If they are different, it signifies the end of a group of identical
 *    characters, so we increment the distinct character counter `x`.
 * 6. After the loop, `x` holds the total count of distinct characters.
 * 7. Check the parity of `x` and print the corresponding message.
 *
 * Note on the loop: The condition `s[i] != s[i+1]` accesses `s[s.size()]` in
 * the last iteration, which is out-of-bounds access and technically undefined
 * behavior. However, in many C++ implementations, `s[s.size()]` returns a null
 * terminator ('\0'), which allows the logic to work correctly by counting the
 * last group of characters. A safer approach would be to use `std::unique`.
 *
 * @complexity
 * - Time Complexity: O(L log L), where L is the length of the string, dominated
 *   by the sorting operation.
 * - Space Complexity: O(L) to store the input string. Some `std::sort`
 *   implementations might use auxiliary space.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  // `x` will count the number of distinct characters.
  int x = 0;
  cin >> s;

  // Sort the string to group identical characters together.
  sort(s.begin(), s.end());

  /**
   * @brief Count distinct characters by finding transitions between different characters.
   */
  for (int i = 0; i < s.size(); i++) {
    // If a character is different from the next one, it's the end of a unique character group.
    // This loop has a potential out-of-bounds read on the last iteration (s[i+1]).
    if (s[i] != s[i + 1]) {
      x++;
    }
  }

  // Check the parity of the distinct character count.
  if (x % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  
  // No explicit return 0; main will implicitly return 0.
}
