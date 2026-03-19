/**
 * @file raw/c5e4efe4-76c3-429d-8d34-4f96a8820131/236A.CPP
 * @brief Determines a social action based on the number of distinct characters in a username.
 * The program counts the number of unique characters in a given string (username).
 * If the count is even, it outputs "CHAT WITH HER!".
 * If the count is odd, it outputs "IGNORE HIM!".
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Sort the string alphabetically. This places all identical characters next to each other.
 * 3. Iterate through the sorted string and count the number of times a character is
 *    different from the one immediately following it. This effectively counts the
 *    number of distinct characters.
 * 4. Check the parity of the distinct character count to make a decision.
 *
 * @time_complexity O(N log N), dominated by the string sorting operation, where N is the length of the string.
 * @space_complexity O(log N) or O(N), depending on the space complexity of the `std::sort` implementation.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int cnt = 0;
  cin >> s;

  // Pre-processing step to group identical characters together.
  sort(s.begin(), s.end());

  /**
   * @block
   * Pre-condition: The string `s` is sorted.
   * Invariant: `cnt` holds the number of distinct characters found in the prefix of the string processed so far.
   * This loop counts the unique characters by comparing adjacent elements.
   */
  for (int i = 0; i < s.length(); i++) {
    // When a character differs from the next, it marks the end of a sequence of identical characters.
    if (s[i] != s[i + 1])
      cnt++;
      // @note This check `s[i + 1]` can read past the end of the string on the last iteration,
      // which is undefined behavior, but often works in competitive programming environments
      // due to how strings are allocated and null-terminated.
  }

  /**
   * @block
   * Determines the output based on the parity of the distinct character count.
   * This is the core logic of the problem's decision-making process.
   */
  if (cnt % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
