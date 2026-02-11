/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct characters in their username.
 * @details This program solves the "Boy or Girl" problem from Codeforces using a common C++ idiom.
 * It counts the unique characters in a given username and outputs "CHAT WITH HER!" if the count
 * is even, and "IGNORE HIM!" if it's odd.
 *
 * @see https://codeforces.com/problemset/problem/236/A
 *
 * Algorithm:
 * 1. Read the input username into a std::string.
 * 2. Sort the string alphabetically. This is a prerequisite for the `std::unique` algorithm.
 * 3. Use `std::unique` to shuffle the unique characters to the front of the string and get an
 *    iterator to the end of this unique range.
 * 4. Erase the redundant characters from the string, leaving only the unique ones.
 * 5. The length of the modified string is the count of distinct characters.
 * 6. Check if this count is even or odd and print the result.
 *
 * Time Complexity: O(N log N), where N is the length of the username, dominated by the `std::sort` operation.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // input: The username string.
  string input;
  // count: Stores the number of distinct characters.
  int count;
  cin >> input;

  // Sort the string to group identical characters together.
  sort(input.begin(), input.end());

  /**
   * @brief Removes duplicate characters using the sort-unique idiom.
   *
   * Block Logic: `std::unique` reorders the input range so that each unique element
   * appears once at the beginning, and returns an iterator to the new end of that range.
   * The string is then resized to this new end, effectively removing duplicates.
   */
  auto result = unique(input.begin(), input.end());
  input.erase(result, input.end());

  // The length of the string now equals the number of unique characters.
  count = input.length();

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * If the count is even, the user is assumed to be female.
   * If the count is odd, the user is assumed to be male.
   */
  if (count % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
