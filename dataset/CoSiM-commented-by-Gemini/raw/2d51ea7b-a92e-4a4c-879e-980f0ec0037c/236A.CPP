/**
 * @file 236A.cpp
 * @brief Solution to the "Boy or Girl" problem on Codeforces.
 * @details This program determines if a user is male or female based on the number of distinct
 * characters in their username. If the number of distinct characters is odd, the user is
 * identified as male ("IGNORE HIM!"). If it's even, the user is female ("CHAT WITH HER!").
 * This implementation uses a `std::unordered_map` to efficiently track unique characters.
 *
 *  Created by Tico on 3/17/17.
 *  Copyright   2017 Thiago Ribeiro. All rights reserved.
 *
 * Algorithm:
 * 1. Read the input username string `s`.
 * 2. Initialize an `std::unordered_map<char, int>` to act as a frequency map or a set.
 * 3. Iterate through each character of the string.
 * 4. For each character, add it to the map. The map's structure automatically handles duplicates;
 *    each unique character will exist as a key only once.
 * 5. After the loop, the size of the map (`h.size()`) equals the number of distinct characters.
 * 6. Use a ternary operator to check the parity of the map's size.
 *    - If the size is even, print "CHAT WITH HER!".
 *    - If the size is odd, print "IGNORE HIM!".
 *
 * Time Complexity: O(N) on average, where N is the length of the string. The average time complexity for insertion into an `std::unordered_map` is O(1).
 * Space Complexity: O(K), where K is the number of unique characters (at most 26 for lowercase English alphabet).
 */
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main() {
  // Use an unordered_map as a set to store unique characters encountered.
  unordered_map<char, int> h;
  string s;

  cin >> s;

  /**
   * Block Logic: Iterate through the string to find all unique characters.
   * Pre-condition: 's' contains the input username.
   * Invariant: 'h' contains each unique character from the processed part of the string as a key.
   */
  for (const char c : s) {
    // If the character is already in the map, do nothing.
    if (h[c] == 1)
      continue;
    // Otherwise, add the character to the map.
    else
      h[c] = 1;
  }

  // Use a ternary operator to select the output based on the parity of the number of unique characters.
  h.size() % 2 == 0 ? cout << "CHAT WITH HER!" : cout << "IGNORE HIM!" << endl;
}
