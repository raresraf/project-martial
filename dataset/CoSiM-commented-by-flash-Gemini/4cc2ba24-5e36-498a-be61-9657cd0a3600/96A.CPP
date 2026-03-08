/**
 * @file 96A.CPP
 * @brief Checks for "dangerous positions" in a string, defined as seven consecutive identical characters ('0' or '1').
 *
 * Algorithm: String searching using `std::string::find` to locate specific patterns.
 * Time Complexity: O(L), where L is the length of the input string. `std::string::find` is linear.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, especially <string> and <iostream>.

using namespace std;

/**
 * @brief Main function to detect dangerous positions in an input string.
 * @pre Input consists of a single string containing '0's and '1's.
 * @post Prints "YES" if a dangerous position is found (seven consecutive '0's or '1's), otherwise "NO".
 */
int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout
  // to optimize high-throughput I/O, which is crucial for competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifndef ONLINE_JUDGE
  // Block Logic: Redirects standard input, output, and error streams to files when not running in an online judge environment.
  // Functional Utility: Facilitates local debugging by allowing input from 'input.txt', output to 'output.txt', and errors to 'error.txt'.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
  freopen("error.txt", "w", stderr);
#endif

  string str; // Variable to store the input string.
  // Functional Utility: Reads the input string from standard input.
  cin >> str;

  /**
   * Block Logic: Checks if the string contains a sequence of seven '1's or seven '0's.
   * Invariant: `string::npos` is returned if the substring is not found.
   *            If either "1111111" or "0000000" is found, it indicates a dangerous position.
   */
  if ((str.find("1111111") != string::npos) || // Checks for seven consecutive '1's.
      (str.find("0000000") != string::npos))   // Checks for seven consecutive '0's.
    cout << "YES"; // Prints "YES" if a dangerous position is found.
  else
    cout << "NO"; // Prints "NO" otherwise.

  // Functional Utility: Prints the execution time to standard error for performance analysis.
  cerr << "time taken: " << (float)clock() / CLOCKS_PER_SEC << " secs" << endl;
  // Indicates successful program execution.
  return 0;
}
