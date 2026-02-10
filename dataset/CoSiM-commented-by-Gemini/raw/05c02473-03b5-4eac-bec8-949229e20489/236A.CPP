/**
 * @file 236A.CPP
 * @brief Solution for competitive programming problem "Boy or Girl".
 * @details This program determines a user's gender based on the number of
 * distinct characters in their username. If the count of distinct characters
 * is odd, the user is assumed to be male ("IGNORE HIM!"). If the count is even,
 * the user is assumed to be female ("CHAT WITH HER!").
 *
 * Algorithm:
 * 1. Read the username as a string.
 * 2. Sort the string alphabetically. This places all identical characters
 *    adjacent to each other.
 * 3. Iterate through the sorted string and count the number of unique
 *    characters by comparing each character to the next one. A new distinct
 *    character is found when `name[i]` is different from `name[i+1]`.
 * 4. Check the parity of the final count of distinct characters.
 * 5. If the count is odd, print "IGNORE HIM!".
 * 6. If the count is even, print "CHAT WITH HER!".
 *
 * Time Complexity: O(N log N) due to the sorting of the string, where N is the
 * length of the username.
 * Space Complexity: O(N) to store the username string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string name;
  // Read the user's name.
  cin >> name;

  int cnt = 0, len = name.length();
  // Sort the string to group identical characters together.
  sort(name.begin(), name.end());
  // Invariant: 'cnt' is the number of distinct characters found so far.
  for (int i = 0; i < len; i++) {
    // If the current character is different from the next one, it's a new
    // distinct character. The loop boundary `i < len` is safe because
    // `name[len]` will be the null terminator, which is different from any
    // character in the sorted name unless the name is empty.
    if (name[i] != name[i + 1])
      cnt++;
  }

  // Determine the output based on the parity of the distinct character count.
  if (cnt % 2 != 0)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
