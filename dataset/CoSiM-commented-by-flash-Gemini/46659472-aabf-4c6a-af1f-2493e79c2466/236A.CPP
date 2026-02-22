/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A: Boy or Girl (Optimized Implementation).
 * @details This program classifies a given username as "boy" or "girl" based on the parity
 * of the number of distinct characters it contains. If the count of distinct characters
 * is even, it's a "girl" username ("CHAT WITH HER!"). If the count is odd, it's a "boy"
 * username ("IGNORE HIM!"). This implementation efficiently counts distinct characters
 * by first sorting the string and then iterating through the sorted string.
 * @algorithm Sorting and linear scan for distinct character counting.
 * Time Complexity: $O(L \log L)$ due to sorting, where L is the length of the username string.
 * Space Complexity: $O(L)$ to store the string.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A: Boy or Girl (Optimized Implementation).
 * @details This program classifies a given username as "boy" or "girl" based on the parity
 * of the number of distinct characters it contains. If the count of distinct characters
 * is even, it's a "girl" username ("CHAT WITH HER!"). If the count is odd, it's a "boy"
 * username ("IGNORE HIM!"). This implementation efficiently counts distinct characters
 * by first sorting the string and then iterating through the sorted string.
 * @algorithm Sorting and linear scan for distinct character counting.
 * Time Complexity: $O(L \log L)$ due to sorting, where L is the length of the username string.
 * Space Complexity: $O(L)$ to store the string.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream, string, algorithm).
using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: orchestrates the username classification logic.
   */
  string s; // @var s: The input username string.
  int x = 0; // @var x: Counter for the number of distinct characters.

  cin >> s; // Reads the input username string.

  /**
   * @brief Sorts the input string to group identical characters together.
   * @details Sorting allows for a simple linear scan to count distinct characters.
   * @param s.begin() Iterator to the beginning of the string.
   * @param s.end()   Iterator to the end of the string.
   */
  sort(s.begin(), s.end());

  /**
   * @brief Counts the number of distinct characters in the sorted string.
   * @block_logic Iterates through the sorted string, comparing adjacent characters.
   * @invariant `x` increments only when a character is different from the next,
   *            effectively counting distinct characters.
   * @note The loop goes up to `s.size()`, but `s[i+1]` might access out of bounds
   *       if `s.size()` is used as the upper bound without considering `i+1`.
   *       However, `s.size() - 1` is usually the correct upper bound for `i` when
   *       accessing `s[i+1]`. Adhering to "Zero Code Mutation" rule, the code is commented as is.
   */
  for (int i = 0; i < s.size(); i++) {
    // Checks if the current character is different from the next.
    // This comparison will detect the start of a new distinct character group.
    // Logic: If s.size() is 1, i=0, loop runs, s[0] != s[1] is false (s[1] is out of bounds or null), x does not increment.
    // If last char: s[i] != s[s.size()] (s[s.size()] is null terminator), x increments. This effectively counts characters + 1.
    // This is likely intended to be `for (int i = 0; i < s.size() - 1; i++)` and then `x` is initialized to 1 if `s` is not empty.
    if (s[i] != s[i + 1]) { 
      x++; // Increment distinct character count.
    }
  }

  /**
   * @brief Classifies the username based on the parity of the distinct character count.
   * @block_logic Checks if `x` (distinct character count) is even or odd.
   * @invariant Prints "CHAT WITH HER!" if `x` is even, otherwise "IGNORE HIM!".
   */
  if (x % 2 == 0)
    cout << "CHAT WITH HER!" << endl; // Output for even number of distinct characters.
  else
    cout << "IGNORE HIM!" << endl;  // Output for odd number of distinct characters.
}
