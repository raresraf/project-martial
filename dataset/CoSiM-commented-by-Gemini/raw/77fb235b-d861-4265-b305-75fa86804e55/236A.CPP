/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from competitive programming platforms.
 *
 * This program determines whether to chat with a user based on their username.
 * The rule is: if the number of distinct characters in the username is even,
 * the user is female ("CHAT WITH HER!"). If the number of distinct characters
 * is odd, the user is male ("IGNORE HIM!").
 *
 * @algorithm
 * The implemented method counts distinct characters by first sorting the string.
 *
 * 1. Read the username string `s`.
 * 2. Sort the string alphabetically. This groups all identical characters together.
 *    For example, "abacaba" becomes "aaaabbc".
 * 3. Initialize a counter `count` to 0.
 * 4. Iterate through the sorted string from the beginning up to the second-to-last
 *    character.
 * 5. In each step, compare the current character `s[i]` with the next character `s[i+1]`.
 * 6. If `s[i]` is different from `s[i+1]`, it signifies a change from one distinct
 *    character to another. Increment `count`.
 * 7. After the loop, `count` will hold the number of distinct characters minus one.
 *    For "aaaabbc", the changes are from 'a' to 'b' and from 'b' to 'c', so `count`
 *    will be 2. The number of distinct characters is `count + 1`, which is 3.
 *
 * 8. Determine the final output based on the parity of the number of distinct characters.
 *    - Number of distinct characters = `count + 1`.
 *    - If `count` is odd, then `count + 1` is even. (Female case)
 *    - If `count` is even, then `count + 1` is odd. (Male case)
 *
 * 9. The code checks `if (count % 2)`.
 *    - This is true if `count` is odd, which means the number of distinct characters
 *      is even. The program correctly prints "CHAT WITH HER!".
 *    - This is false if `count` is even, which means the number of distinct characters
 *      is odd. The program correctly prints "IGNORE HIM!".
 *
 * @alternative_algorithm
 * A more direct approach would be to insert all characters of the string into a
 * `std::set` and then check the parity of `set.size()`.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the string, dominated by the sorting operation.
 * Space Complexity: O(L) to store the string, or O(1) if the input string is modified in-place.
 */
#include <bits/stdc++.h>

using namespace std;

void solve() {
  string s;
  cin >> s;
  
  // `count` will store the number of transitions between different characters.
  // This will be equal to (number of distinct characters - 1).
  int count = 0;

  // Sort the string to group identical characters together.
  sort(s.begin(), s.end());

  /**
   * @brief Count the number of times adjacent characters are different.
   */
  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] != s[i + 1])
      count++;
  }

  // The number of distinct characters is `count + 1`.
  // The program needs to check if `count + 1` is even or odd.
  // `(count + 1) % 2 == 0` is for "CHAT WITH HER!" (female)
  // `(count + 1) % 2 != 0` is for "IGNORE HIM!" (male)
  // The logic below is equivalent.
  if (count % 2) {
    // If `count` is odd, `count + 1` is even -> Female.
    cout << "CHAT WITH HER!" << endl;
  } else {
    // If `count` is even, `count + 1` is odd -> Male.
    cout << "IGNORE HIM!" << endl;
  }
}

int main() {
  solve();
  return 0;
}
