/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces (Problem 236A).
 *        This program reads a username (string) and determines if the number
 *        of unique characters in it is odd or even. Based on this, it prints
 *        "IGNORE HIM!" or "CHAT WITH HER!", respectively.
 * Algorithm: Character Counting (Unique) / Set Usage
 * Time Complexity: O(L log U), where L is the length of the input string and U
 *                  is the number of unique characters. `log U` comes from `std::set::insert`.
 *                  In the worst case (e.g., all unique characters up to 26 for English alphabet),
 *                  this is effectively O(L).
 * Space Complexity: O(U) to store unique characters in the `std::set`. In the
 *                   worst case, this is O(1) if the character set size is bounded (e.g., 26).
 */

#include <iostream> // Required for input/output operations (cin, cout).
#include <set>      // Required for using std::set to store unique characters.
#include <string>   // Required for using std::string.

/*
 * Contest: Code Forces Round #146 (Task 236-A)
 * URL: http://codeforces.ru/contest/236/problem/A
 */

/**
 * @brief Main function to read a username, count its unique characters, and print a message.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 if the program executes successfully.
 */
int main(int argc, char **argv) {
  // Declare a set to store unique characters. std::set automatically handles uniqueness.
  std::set<int> ch;

  std::string s; // Declare a string to store the username.
  // Read the username from standard input.
  std::cin >> s;

  /**
   * Block Logic: Iterates through each character of the input string and inserts it into the set.
   * Invariant: After each iteration, `ch` contains all unique characters encountered so far.
   */
  for (auto it = s.cbegin(); it != s.cend(); ++it) {
    ch.insert(*it); // Insert the current character into the set. Duplicates are ignored by std::set.
  }

  // Block Logic: Determines the output message based on the parity of the unique character count.
  // Pre-condition: `ch.size()` holds the number of unique characters.
  // Post-condition: "IGNORE HIM!" is printed if the count is odd, "CHAT WITH HER!" if even.
  // The expression `ch.size() & 1` is a bitwise check for oddness (result is 1 if odd, 0 if even).
  std::cout << ((ch.size() & 1) ? "IGNORE HIM!" : "CHAT WITH HER!")
            << std::endl;
  return 0; // Indicate successful program termination.
}
