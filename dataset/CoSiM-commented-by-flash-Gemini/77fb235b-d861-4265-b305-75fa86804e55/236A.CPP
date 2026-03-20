/**
 * @file 236A.CPP
 * @brief Solution to a string manipulation problem (Codeforces problem 236A - Boy or Girl).
 * Algorithm: Counts the number of distinct characters in an input string.
 *            The string is first sorted to group identical characters, simplifying the counting process.
 *            Then, it iterates through the sorted string, incrementing a counter whenever a character differs from its successor.
 *            Finally, it determines the output based on the parity of the distinct character count.
 * Time Complexity: O(N log N) due to sorting, where N is the length of the input string.
 * Space Complexity: O(N) for storing the string (or O(1) if in-place sort is considered and string modifications are allowed).
 */
#include <bits/stdc++.h>

using namespace std;

void solve() {
  /**
   * @brief Solves the "Boy or Girl" problem.
   *
   * Reads a string (username), counts the number of distinct characters in it.
   * Based on the parity (odd/even) of this count, it prints
   * "CHAT WITH HER!" (odd) or "IGNORE HIM!" (even).
   *
   * This logic is specific to the problem statement for Codeforces 236A.
   */
  string s;
  cin >> s;
  int count = 0;

  // Block Logic: Sorts the input string to group identical characters together.
  // Pre-condition: 's' contains the input username.
  // Post-condition: Characters in 's' are arranged in non-decreasing order.
  sort(s.begin(), s.end());

  // Block Logic: Iterates through the sorted string to count distinct adjacent characters.
  // Invariant: 'count' stores the number of times a character differs from its next character,
  //            effectively counting distinct characters (minus one for the first occurrence).
  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] != s[i + 1])
      count++;
  }

  // Block Logic: Determines the output message based on the parity of the distinct character count.
  // Pre-condition: 'count' holds the number of transitions between distinct characters.
  // Post-condition: Prints "CHAT WITH HER!" if 'count' is odd, "IGNORE HIM!" if 'count' is even.
  if (count % 2) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
}

int main() {
  /**
   * @brief Main function of the program.
   *
   * Calls the `solve` function to execute the problem's logic.
   * This structure is common in competitive programming to keep `main` concise.
   * @return int Returns 0 upon successful execution.
   */
  solve();
  return 0;
}
