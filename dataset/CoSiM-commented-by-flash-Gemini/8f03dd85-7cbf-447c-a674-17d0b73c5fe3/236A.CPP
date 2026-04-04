/**
 * @file 236A.CPP
 * @brief Determines a message based on the parity of unique characters in an input string.
 *
 * This program reads a single string and counts the number of distinct characters within it.
 * If the count of unique characters is odd, it outputs "IGNORE HIM!".
 * If the count is even, it outputs "CHAT WITH HER!".
 *
 * Algorithm: Character frequency counting using a `std::set` to store unique characters.
 * Time Complexity: O(N log K), where N is the length of the input string and K is the
 *                  number of unique characters (K <= 26 for English alphabet). The `log K`
 *                  factor comes from `std::set::insert` operation.
 * Space Complexity: O(K) for storing unique characters in the set.
 */
#include <bits/stdc++.h>

// Functional Utility: Defines 'll' as a shorthand for 'long long int' for convenience in competitive programming.
#define ll long long int

// Functional Utility: Defines 'mod' as a commonly used large prime number for modulo operations.
#define mod 1000000007

// Functional Utility: Defines 'pb' as a shorthand for 'push_back' for vector operations.
#define pb push_back

// Functional Utility: Defines 'mp' as a shorthand for 'make_pair' for creating pairs.
#define mp make_pair

// Functional Utility: Defines 'ff' as a shorthand for 'first' for accessing the first element of a pair/tuple.
#define ff first

// Functional Utility: Defines 'ss' as a shorthand for 'second' for accessing the second element of a pair/tuple.
#define ss second

// Functional Utility: Defines 'all(v)' as a shorthand to get iterators to the beginning and end of a container 'v'.
#define all(v) v.begin(), v.end()

// Functional Utility: Uses the standard namespace to avoid repeatedly typing 'std::'.
using namespace std;

int main()

{
  /**
   * @brief Main function: Reads a string, counts unique characters, and prints a message.
   * @return 0 if the program executes successfully.
   */

  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  // 'ios_base::sync_with_stdio(false)' unties C++ streams from C standard streams.
  // 'cin.tie(0)' prevents 'cin' from flushing 'cout' before each input operation.
  ios_base::sync_with_stdio(false);

  cin.tie(0);

  string s;

  cin >> s; // Read the input string.

  std::set<char> d; // Declare a set to store unique characters.
  /**
   * @brief Populates the set 'd' with unique characters from the input string 's'.
   * @pre 's' contains the input string.
   * @post 'd' contains all distinct characters from 's'.
   * @invariant Each character from 's' is processed once, and duplicates are automatically
   *            handled by the set's properties.
   */
  for (auto i : s) // Iterate over each character in the string 's'.
    d.insert(i);   // Insert the character into the set 'd'.

  /**
   * @brief Determines the output message based on the parity of the number of unique characters.
   * @pre 'd.size()' holds the count of unique characters.
   * @invariant The condition 'd.size() & 1' checks if the size is odd (bitwise AND with 1).
   */
  if (d.size() & 1) // If the number of unique characters is odd.
    cout << "IGNORE HIM!";
  else // If the number of unique characters is even.
    cout << "CHAT WITH HER!";

  return 0;
}