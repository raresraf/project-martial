/**
 * @file 236A.CPP
 * @brief Solution for Codeforces problem 236A, "Boy or Girl".
 *
 * @description
 * This program determines a user's gender based on the number of distinct
 * characters in their username. If the count of distinct characters is even,
 * the user is identified as female ("CHAT WITH HER!"). If the count is odd,
 * the user is identified as male ("IGNORE HIM!").
 *
 * @algorithm
 * The program uses the "sort-unique" C++ idiom to count distinct characters.
 * 1. Read the username string.
 * 2. Sort the string alphabetically. This groups identical characters together.
 * 3. Use `std::unique` to shuffle the unique elements to the front of the
 *    string and get an iterator to the end of the unique range.
 * 4. Create a new string containing only the unique elements.
 * 5. Check if the length of this new string (the count of distinct characters)
 *    is even or odd.
 * 6. Print the corresponding message.
 *
 * @time_complexity O(L log L), where L is the length of the username. The
 *   complexity is dominated by the sorting step.
 * @space_complexity O(L) to store the username string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  cin >> a;

  // Sort the string to group identical characters together.
  // For example, "hello" becomes "ehllo".
  sort(a.begin(), a.end());

  // `std::unique` rearranges the string so that each unique character appears
  // once at the beginning and returns an iterator to the end of this unique range.
  // For "ehllo", this would result in "ehlo" followed by unspecified characters.
  auto r = unique(a.begin(), a.end());

  // Create a new string from the unique character range.
  a = string(a.begin(), r);

  // Block Logic: Determine the output based on the parity of the distinct character count.
  // `a.length()` now holds the number of unique characters.
  a = a.length() % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!";
  cout << a << endl;
}
