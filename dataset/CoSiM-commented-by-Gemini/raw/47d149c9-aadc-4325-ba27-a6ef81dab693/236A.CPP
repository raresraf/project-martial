/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" programming contest problem.
 * @details The program determines a user's gender based on their username. If the
 * number of distinct characters in the name is even, the user is female
 * ("CHAT WITH HER!"). If the number of distinct characters is odd, the user
 * is male ("IGNORE HIM!").
 *
 * Algorithm:
 * 1. Read the username string.
 * 2. Count the number of unique characters in the string. This implementation
 *    achieves this by first sorting the string, which makes all identical
 *    characters adjacent.
 * 3. It then iterates through the sorted string, incrementing a counter each
 *    time a character differs from the one next to it.
 * 4. Finally, it checks the parity of the counter. An even count results in
 *    "CHAT WITH HER!", and an odd count results in "IGNORE HIM!".
 *
 * Note on Implementation: The loop for counting distinct characters has a potential
 * out-of-bounds memory access on its final iteration (`s[i + 1]`). A more robust
 * approach would use `std::set` or the `std::unique` algorithm.
 *
 * Time Complexity: O(L log L), where L is the length of the string, dominated by the sorting step.
 * Space Complexity: O(L) to store the string.
 */
#include <algorithm>
#include <iostream>

using namespace std;
int main() {

  string s;
  int a; // Unused variable.
  cin >> s;

  // Sort the string to group identical characters together.
  sort(s.begin(), s.end());
  int count = 0;
  
  /**
   * @block
   * @description Counts the number of distinct characters by iterating through the
   * sorted string and incrementing the count when a character changes.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != s[i + 1]) {
      count++;
    }
  }
  
  /**
   * @block
   * @description Determines the output based on whether the number of distinct
   * characters is even or odd.
   */
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}