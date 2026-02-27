/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces.
 * @details The problem determines a user's gender based on their username.
 * If the number of distinct characters in the username is even, the user is
 * female ("CHAT WITH HER!"). If odd, the user is male ("IGNORE HIM!").
 *
 * @algorithm
 * This solution uses a `std::map` to efficiently count the number of
 * distinct characters.
 * 1. Read the input username string.
 * 2. Create a `std::map` to store character frequencies. A map automatically
 *    handles unique keys, so its final size will equal the number of
 *    distinct characters.
 * 3. Iterate through the string, incrementing the map entry for each character.
 *    `map[character]++` is a concise way to build this frequency map.
 * 4. After the loop, the number of distinct characters is simply `map.size()`.
 * 5. Check if the size is even or odd. This is done efficiently using a
 *    bitwise AND operation (`size & 1`). An odd number will result in 1 (true),
 *    an even number in 0 (false).
 * 6. Print the corresponding message using a ternary operator.
 *
 * @complexity
 * Time Complexity: O(L log K), where L is the length of the string and K is the
 * number of unique characters (at most 26 for this problem). Each map access
 * takes O(log K) time.
 * Space Complexity: O(K) to store the unique characters in the map.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  string s;
  cin >> s;

  // Use a map to count the frequency of each character.
  // The number of unique keys in the map will be the number of distinct characters.
  map<char, int> m;

  /**
   * @block
   * @brief Populate the frequency map.
   * For each character in the string, its count in the map is incremented.
   */
  for (int i = 0; i < (int)s.length(); i++) {
    m[s[i]]++;
  }

  /**
   * @brief Determine output based on the parity of the number of distinct characters.
   * `m.size()` gives the count of distinct characters.
   * `& 1` is a bitwise AND operation to check if the count is odd (result 1) or even (result 0).
   */
  cout << (m.size() & 1 ? "IGNORE HIM!" : "CHAT WITH HER!");
}
