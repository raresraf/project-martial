/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces contest 71.
 *
 * @details
 * The problem requires abbreviating words that are longer than 10 characters.
 * The abbreviation is formed by taking the first and last characters of the word
 * and placing the count of the letters between them. For instance, "localization"
 * becomes "l10n". Words with 10 or fewer characters remain unchanged.
 *
 * The program reads an integer `t` representing the number of words to process.
 * It then enters a loop that runs `t` times. In each iteration, it reads a word,
 * checks its length, and prints either the original word or its abbreviation
 * based on the problem's criteria.
 *
 * Algorithm:
 * 1. Read the number of test cases, `t`.
 * 2. Loop `t` times:
 *    a. Read the input word as a string.
 *    b. Check if the length of the string is greater than 10.
 *    c. If it is, construct the abbreviated string:
 *       - Start with the first character of the original string.
 *       - Append the string representation of (length - 2).
 *       - Append the last character of the original string.
 *       - Print the new abbreviated string.
 *    d. If the length is not greater than 10, print the original string.
 *
 * Time Complexity: O(t * L) where `t` is the number of test cases and `L` is the length of the string.
 * Since converting the length to a string takes time proportional to the number of digits in the length,
 * which is log(L), the complexity for each word is dominated by reading the string, making it O(L).
 * Thus, the total complexity is O(t * L_max) where L_max is the maximum length of any word.
 *
 * Space Complexity: O(L_max) to store the longest input word.
 */
#include <bits/stdc++.h> // Includes all standard C++ libraries
using namespace std;
#define M 1000000007 // A large prime number, often used for modulo operations in competitive programming
typedef long long ll; // Type alias for long long

/**
 * @brief Configures I/O for performance and local testing.
 *
 * This function is designed to be called at the beginning of `main`. It performs two main tasks:
 * 1. If the code is compiled without the `ONLINE_JUDGE` macro (common in local test environments),
 *    it redirects standard input and standard output to files ("peepee.txt" and "poopoo.txt").
 * 2. It optimizes I/O performance by un-syncing C++ iostreams from C stdio and un-tying `cin` from `cout`.
 */
void gz() {

#ifndef ONLINE_JUDGE
  // Redirects I/O to files for easier testing and debugging on a local machine.
  freopen("peepee.txt", "r", stdin); // Redirect stdin from peepee.txt
  freopen("poopoo.txt", "w", stdout); // Redirect stdout to poopoo.txt
#endif

  // Functional Utility: Speeds up I/O operations by decoupling C++ streams from the underlying C standard streams.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL); // Unties cin from cout to avoid unnecessary flushes.
  cout.precision(10); // Sets the default floating-point precision for cout.
}

// int p = 31; // A commented out variable, likely used for hashing in a different problem

int main(int argc, char const *argv[]) {
  gz(); // Call the I/O setup function.
  {
    int t; // Variable to store the number of words to process.
    cin >> t;
    // Block Logic: Processes each word according to the problem statement.
    // Invariant: The loop will execute 't' times.
    while (t--) {
      string s; // Variable to store the input word.
      cin >> s;
      // Pre-condition: Check if the word's length exceeds the threshold for abbreviation.
      if (s.length() > 10) {
        string x; // String to build the abbreviation.
        // Inline: Append the first character of the word.
        x += s[0];
        // Inline: Append the count of characters between the first and last.
        x += to_string(s.length() - 2);
        // Inline: Append the last character of the word.
        x += s[s.length() - 1];
        cout << x << endl; // Output the resulting abbreviation.
      } else
        // If the word is not too long, print it as is.
        cout << s << endl;
    }
  }
  return 0; // Indicates successful program termination.
}
