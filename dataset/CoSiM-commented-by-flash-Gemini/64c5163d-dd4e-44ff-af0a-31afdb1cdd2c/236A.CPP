/**
 * @64c5163d-dd4e-44ff-af0a-31afdb1cdd2c/236A.CPP
 * @brief This program solves problem 236A, "Boy or Girl".
 * It reads a username (string) and determines if the number of distinct characters
 * in the username is odd or even. Based on this, it prints a specific message:
 * "IGNORE HIM!" if the count is odd, and "CHAT WITH HER!" if the count is even.
 *
 * Algorithm: Naive character counting. It uses a nested loop to find distinct characters,
 * which is less efficient than using a hash set (like `std::set` or `std::unordered_set`)
 * or a boolean frequency array.
 * Time Complexity: O(L^2) where L is the length of the string, due to the nested loops.
 * Space Complexity: O(1) beyond input string storage (for few integer variables).
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers.
#include <string>        // Explicitly included for std::string.
#define ll long long     // @brief Defines 'll' as a shorthand for 'long long' (unused in this solution).
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to execute the "Boy or Girl" problem logic.
 * Reads a username, counts its distinct characters using a nested loop,
 * and prints a message based on the parity of the distinct count.
 * @return 0 on successful execution.
 */
int main() {

// Block Logic: Conditional compilation for local file I/O during development/testing.
// This block is active only when `ONLINE_JUDGE` is not defined (i.e., local environment).
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin); // Redirects standard input from "input.txt".
  freopen("output.txt", "w", stdout); // Redirects standard output to "output.txt".
#endif
  
  string a; // Declares a string variable to store the input username.
  int pas = 1; // 'pas' (likely 'distinct_count' or 'unique_chars') initialized to 1.
               // This implies the first character is considered unique initially.
  int num; // 'num' (likely 'duplicate_count' or 'found_in_previous') to count duplicate characters.
  cin >> a; // Reads the username from standard input.
  
  /**
   * Block Logic: Iterates through the input string to count distinct characters.
   * For each character, it checks against all preceding characters to see if it's a duplicate.
   * If a character is unique (not found among previous characters), `pas` is incremented.
   * Invariant: `pas` eventually holds the count of distinct characters in the string.
   */
  for (int i = 0; i < a.size(); i++) { // Outer loop iterates through each character.
    num = 0; // Resets duplicate counter for each new character.
    for (int j = 0; j < i; j++) { // Inner loop checks against preceding characters.
      if (a[i] == a[j]) { // Inline: If the current character is a duplicate of a previous one.
        num++; // Increments the duplicate counter.
      }
    }
    if (num == 0) { // If `num` remains 0, it means the character 'a[i]' is unique so far.
      pas++; // Increments the count of distinct characters.
    }
    num = 0; // Resets `num` for the next outer loop iteration (redundant after `if (num == 0)`).
  }
  
  /**
   * Block Logic: Checks if the number of distinct characters is odd or even and prints the corresponding message.
   * The problem's logic requires `pas` to represent the *actual* count of distinct characters,
   * but its initialization to 1 and subsequent incrementing means it effectively overcounts by 1.
   * The final check `pas % 2 == 0` corresponds to whether the *true* distinct count is odd (since `pas` is true_count + 1).
   * Invariant: One of "IGNORE HIM!" or "CHAT WITH HER!" is printed.
   */
  if (pas % 2 == 0) { // If (distinct_count + 1) is even, then distinct_count is odd.
    cout << "IGNORE HIM!"; // Prints "IGNORE HIM!".
  } else { // If (distinct_count + 1) is odd, then distinct_count is even.
    cout << "CHAT WITH HER!"; // Prints "CHAT WITH HER!".
  }
  return 0; // Indicates successful program termination.
}
