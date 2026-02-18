/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem (236A) from Codeforces.
 * @details Determines if a user is male or female based on their username. If the number of
 * distinct characters in the username is even, the user is female ("CHAT WITH HER!").
 * If it's odd, the user is male ("IGNORE HIM!").
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution counts the number of distinct characters in the input string
 * by sorting the string and then removing adjacent duplicates.
 * 1. Read the username string.
 * 2. Sort the string alphabetically. This groups all identical characters together.
 * 3. Use the `std::unique` function to move all unique elements to the beginning of the
 *    string and get an iterator to the new logical end.
 * 4. Use `std::string::erase` to remove the duplicate elements from the end of the string.
 * 5. After this, the length of the string is equal to the number of distinct characters.
 * 6. Check if this new length is even or odd and print the corresponding message.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the string. This is dominated by the
 * `std::sort` operation.
 * Space Complexity: O(L) to store the string. The operations are performed in-place.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s;
  cin >> s;

  /**
   * @block
   * @brief This block counts the number of distinct characters in the string `s`.
   * @details It uses the standard sort-unique-erase idiom on the string.
   */
  sort(s.begin(), s.end());
  s.erase(unique(s.begin(), s.end()), s.end());

  /**
   * @block
   * @brief Checks the parity of the distinct character count to determine the output.
   */
  if ((s.length()) % 2 == 0)
    cout << "CHAT WITH HER!\n";

  else
    cout << "IGNORE HIM!\n";
}
