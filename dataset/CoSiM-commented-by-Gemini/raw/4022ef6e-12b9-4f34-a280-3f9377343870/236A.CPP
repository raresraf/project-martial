/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem's decision logic is based on the number of distinct characters in a
 * given username. If the count of distinct characters is odd, the user is
 * identified as male ("IGNORE HIM!"). If the count is even, the user is female
 * ("CHAT WITH HER!").
 *
 * This implementation uses `std::set` to efficiently count the distinct characters:
 * 1. Read the username string.
 * 2. Create an empty `std::set<char>`.
 * 3. Iterate through the input string and insert each character into the set.
 *    `std::set` automatically handles uniqueness, so duplicates are ignored.
 * 4. After the loop, the size of the set (`distinctCharacters.size()`) equals the
 *    number of distinct characters in the username.
 * 5. Check if this size is odd or even using the modulo operator.
 * 6. Print the appropriate message based on the parity (odd/even) of the count.
 *
 * @complexity
 * - Time Complexity: O(L * log(K)), where L is the length of the string and K is
 *   the number of distinct characters (at most 26 for lowercase English letters,
 *   so effectively constant time for insertion). The overall complexity is dominated
 *   by iterating through the string, making it approximately O(L).
 * - Space Complexity: O(K), to store the distinct characters in the set.
 *   In the context of this problem, K is at most the size of the character set (e.g., 26),
 *   so space complexity is effectively O(1).
 */
#include <iostream>
#include <set>

using namespace std;

int main() {
  string userName;
  set<char> distinctCharacters;

  getline(cin, userName);
  /**
   * @brief Populates a set with the characters from the username.
   * The size of the set will then represent the number of distinct characters.
   */
  for (int i = 0; i < userName.size(); i++) {
    distinctCharacters.insert(userName[i]);
  }

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * If the count is odd, output "IGNORE HIM!".
   * If the count is even, output "CHAT WITH HER!".
   */
  if (distinctCharacters.size() % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";

  return 0;
}
