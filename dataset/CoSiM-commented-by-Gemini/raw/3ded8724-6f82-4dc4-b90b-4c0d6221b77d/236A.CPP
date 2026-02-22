/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem (Codeforces 236A).
 *
 * @algorithm
 * The problem is to determine a user's gender based on their username. The rule
 * is: if the number of distinct characters in the username is odd, the user is
 * male ("IGNORE HIM!"); if it's even, the user is female ("CHAT WITH HER!").
 *
 * This solution uses a hash map (`std::unordered_map`) to efficiently count the
 * number of distinct characters:
 * 1. Read the username into a string `s`.
 * 2. Create an `unordered_map` to store character frequencies.
 * 3. Iterate through the input string. For each character, use it as a key in
 *    the map and increment its associated value. The map automatically handles
 *    the creation of new keys for unseen characters.
 * 4. After the loop, the number of distinct characters is equal to the `size()`
 *    of the map.
 * 5. Check the parity (even or odd) of the map's size and print the
 *    corresponding message.
 *
 * @complexity
 * - Time Complexity: O(N) on average, where N is the length of the string,
 *   as each character is inserted into the hash map once.
 * - Space Complexity: O(K), where K is the number of distinct characters
 *   (at most 26 for lowercase English letters). This can be considered O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  // Use an unordered_map to count the frequency of each character.
  // The number of keys in the map will equal the number of distinct characters.
  unordered_map<char, int> mp;
  /**
   * @brief Populate the map with character counts.
   * For each character in the string, `mp[s[i]]++` will either create a new
   * entry for an unseen character or increment the count for an existing one.
   */
  for (int i = 0; i < s.size(); i++) {
    mp[s[i]]++;
  }
  // The size of the map gives the number of distinct characters.
  int n = mp.size();
  /**
   * @brief Check the parity of the number of distinct characters to determine
   * the output according to the problem statement.
   */
  if (n % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}
