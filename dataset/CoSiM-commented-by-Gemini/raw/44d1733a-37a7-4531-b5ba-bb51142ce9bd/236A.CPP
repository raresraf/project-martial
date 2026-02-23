/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem's decision is based on the number of distinct characters in a
 * username. If the count of distinct characters is odd, the user is male
 * ("IGNORE HIM!"). If it's even, the user is female ("CHAT WITH HER!").
 *
 * This solution uses a `std::set<char>` to efficiently count the distinct
 * characters. A `std::set` automatically handles uniqueness, so by inserting
 * all characters of the input string into the set, the final size of the set
 * will be the number of distinct characters.
 *
 * The algorithm is as follows:
 * 1. Read the input username character by character.
 * 2. Insert each character into a `std::set<char>`.
 * 3. After processing the whole input, get the size of the set.
 * 4. Check if the size is even or odd and print the corresponding message.
 *
 * @complexity
 * - Time Complexity: O(L * log(U)), where L is the length of the string and U
 *   is the number of unique characters (at most 26 for this problem). Each
 *   insertion into the set takes O(log U) time.
 * - Space Complexity: O(U) to store the unique characters in the set.
 */
#include <bits/stdc++.h>

// Macro to disable synchronization with C stdio and untie cin/cout,
// for faster I/O in competitive programming.
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  // The `_` macro is invoked to speed up I/O.
  _ 
  
  string s; // Declared but not used, as input is read char by char into `x`.
  
  // m: A set to store the distinct characters from the input username.
  set<char> m;
  char x;

  /**
   * Block Logic: Read characters from standard input and insert them into the set.
   * The set automatically handles storing only the unique characters.
   * The loop `while(cin >> x)` will read non-whitespace characters one by one.
   * For a single-word username, this correctly reads all characters.
   * The newline check is unlikely to be triggered with typical competitive programming input.
   */
  while (cin >> x) {
    if (x == '\n')
      break;
    m.insert(x);
  }

  /**
   * Block Logic: Determine the output based on the parity of the distinct character count.
   * `m.size()` gives the number of unique characters.
   */
  if (m.size() % 2 == 0) {
    cout << "CHAT WITH HER!\n";
  } else {
    cout << "IGNORE HIM!\n";
  }

  return 0;
}