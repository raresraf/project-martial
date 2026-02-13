/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem.
 * @details The program determines if a user is male or female based on the number of
 * distinct characters in their username. If the count of distinct characters is even,
 * the user is identified as female ("CHAT WITH HER!"). If the count is odd, the user
 * is identified as male ("IGNORE HIM!").
 *
 * @section algorithm Algorithm
 * 1. Read the username string from standard input.
 * 2. Count the number of unique characters in the string using the `countUnique` helper function.
 *    - The function first sorts the string, which groups identical characters together.
 *    - It then uses `std::unique` to reorder the string in-place, moving all unique
 *      characters to the beginning and returning an iterator to the end of this unique range.
 *    - The number of unique characters is calculated as the distance from the beginning
 *      of the string to this new end iterator.
 * 3. In `main`, check if the returned count of unique characters is even.
 * 4. If the count is even, print "CHAT WITH HER!". Otherwise, print "IGNORE HIM!".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N log N), where N is the length of the username string. This is dominated by the `std::sort` operation.
 * - Space Complexity: O(N) due to the string copy in `countUnique`, or O(1) if considering only auxiliary space beyond the input string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Counts the number of unique characters in a string.
 * @param str The input string. It is passed by value, so the original string in main is not modified.
 * @return The integer count of distinct characters in the string.
 */
int countUnique(string str) {
  // Sort the string to group identical characters consecutively.
  sort(str.begin(), str.end());
  // `unique` rearranges the string so that each unique character appears once at the
  // beginning of the range. It returns an iterator to the end of the unique range.
  auto i = unique(str.begin(), str.end());
  // The distance between the beginning of the string and the iterator 'i' gives
  // the count of unique characters.
  return i - str.begin();
}

int main() {
  string str;
  cin >> str;
  // The decision logic is based on the parity of the number of distinct characters.
  if (countUnique(str) % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
