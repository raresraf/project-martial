/**
 * @file 236A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Boy or Girl").
 * This program determines a user's gender based on their username. If the number
 * of distinct characters in the username is even, the user is identified as
 * female ("CHAT WITH HER!"). If it's odd, the user is male ("IGNORE HIM!").
 *
 * @algorithm The algorithm uses a `std::set` to efficiently find the number of
 * unique characters in the input string. The size of the set is then checked
 * for parity (even or odd) to determine the output.
 *
 * @time_complexity O(N * log K), where N is the length of the string and K is the
 * number of distinct characters. The log factor is due to the insertion
 * time complexity of `std::set`.
 * @space_complexity O(K) to store the distinct characters in the set.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  string username;
  cin >> username;
  set<char> letters; // A set to store the unique characters.

  /**
   * @block
   * @brief Populates the set with characters from the username.
   * The `std::set` automatically handles duplicates; inserting an existing
   * character has no effect.
   */
  for (auto i = 0; i < username.length(); i++) {
    letters.insert(username[i]);
  }

  /**
   * @block
   * @brief Checks the parity of the distinct character count.
   * `letters.size()` gives the number of unique characters.
   */
  if (letters.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}