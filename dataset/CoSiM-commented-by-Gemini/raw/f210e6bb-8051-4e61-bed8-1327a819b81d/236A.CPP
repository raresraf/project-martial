/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct characters in their name.
 *
 * This program solves the "Boy or Girl" problem. It reads a username as a string.
 * The core logic is to count the number of unique characters in the string.
 * If the count of unique characters is even, the program outputs "CHAT WITH HER!".
 * If the count is odd, it outputs "IGNORE HIM!".
 *
 * @author f210e6bb-8051-4e61-bed8-1327a819b81d
 *
 * @algorithm Character Frequency Counting
 * @complexity
 * Time: O(N log K), where N is the length of the string and K is the number of unique
 *       characters (at most 26 for lowercase English letters). The `log K` factor comes
 *       from insertions into the std::map.
 * Space: O(K), to store the map of unique characters.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // `s` will store the input username.
  string s;
  cin >> s;
  // `in` is a map used to store the frequency of each character in the string.
  // The number of unique characters will be the size of the map.
  map<char, int> in;

  /**
   * @brief Loop to populate the character frequency map.
   * It iterates through each character of the input string `s`.
   */
  for (int i = 0; i < s.length(); i++) {
    // For each character `s[i]`, the map `in` increments its count.
    // If the character is not yet in the map, it's added.
    in[s[i]]++;
  }

  // `x` stores the number of unique characters, which is the size of the map.
  int x = in.size();

  /**
   * @brief Check if the number of unique characters is even or odd.
   * This is the main condition of the problem to determine the output.
   */
  if (x % 2 == 0)
    // If even, the user is identified as female.
    cout << "CHAT WITH HER!" << endl;
  else
    // If odd, the user is identified as male.
    cout << "IGNORE HIM!" << endl;
  return 0;
}