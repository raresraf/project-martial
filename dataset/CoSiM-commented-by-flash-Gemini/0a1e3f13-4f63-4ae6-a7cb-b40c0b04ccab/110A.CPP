/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A - Nearly Lucky Number.
 *
 * This problem defines "lucky digits" as 4 and 7. A "nearly lucky" number is one
 * where the count of its lucky digits is itself a lucky number. The task is to
 * determine if a given input number is "nearly lucky".
 *
 * Algorithm: Digit counting and recursive "lucky" check.
 * The code first counts the occurrences of digits '4' and '7' in the input number.
 * Then, it checks if this count is itself a "lucky" number (i.e., composed only of '4' and '7').
 *
 * Time Complexity: O(log10(N)), where N is the input number.
 * Space Complexity: O(log10(N)) for string conversion.
 */

/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope, simplifying code by avoiding explicit std:: prefixes.

typedef long long LL; // Type definition for long long.
typedef vector<int> vi; // Type definition for vector of integers.
typedef vector<long long> vl; // Type definition for vector of long longs.

#define MOD 1000000007 // Modulo constant, often used in modular arithmetic.
#define _for(i, n) for (int i = 0; i < n; i++) // Loop macro for 0 to n-1.
#define __for(i, n) for (int i = 1; i <= n; i++) // Loop macro for 1 to n.
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++) // Loop macro for s to n.
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++) // Loop macro for 0 to n-1.
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--) // Reverse loop macro for e down to s.
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++) // Iterator loop macro.
#define pb push_back // Macro for vector push_back.
#define mp make_pair // Macro for creating pairs.
#define fi first // Macro for accessing pair's first element.
#define se second // Macro for accessing pair's second element.
#define ALL(x) x.begin(), x.end() // Macro for getting iterators to begin and end of a container.

/**
 * @brief Checks if a given integer consists only of "lucky digits" (4 and 7).
 * Functional Utility: Determines if a number is "lucky" based on its decimal representation.
 * Algorithm: Converts the integer to a string and then checks if any digit other
 * than '4' or '7' is present.
 * Time Complexity: O(log10(a)) for string conversion and iteration.
 * Space Complexity: O(log10(a)) for string storage.
 * @param a The integer to check.
 * @return `true` if `a` is a lucky number, `false` otherwise.
 */
bool isLucky(int a) {
  // Block Logic: Convert integer to string and check for non-lucky digits.
  // Precondition: `a` is a non-negative integer.
  string s_a = to_string(a); // Convert integer to string.
  if (s_a.find("1") != string::npos || // Check if '1' is present.
      s_a.find("2") != string::npos || // Check if '2' is present.
      s_a.find("3") != string::npos || // Check if '3' is present.
      s_a.find("5") != string::npos || // Check if '5' is present.
      s_a.find("6") != string::npos || // Check if '6' is present.
      s_a.find("8") != string::npos || // Check if '8' is present.
      s_a.find("9") != string::npos || // Check if '9' is present.
      s_a.find("0") != string::npos) // Check if '0' is present.
    return false; // If any non-lucky digit found, return false.
  return true; // If only lucky digits found, return true.
}

/**
 * @brief Main function: Solves the Nearly Lucky Number problem.
 * Functional Utility: Reads a large integer `n`, counts its lucky digits (4 and 7),
 * and then determines if this count itself is a lucky number.
 * Algorithm: Two-step process: digit counting, then lucky check on the count.
 */
int main() {
  long long int n; // Declares a long long integer `n` to store the input number.
  cin >> n; // Reads the input number `n`.
  int count = 0; // Initializes a counter for lucky digits.
  string s = to_string(n); // Converts the input number `n` to a string for digit iteration.
  
  // Block Logic: Iterate through the digits of the number `n` to count lucky digits.
  // Precondition: `s` is a valid string representation of `n`.
  // Invariant: `count` accumulates the number of '4's and '7's found so far.
  for (int i = 0; i < s.size(); i++) {
    if (s.at(i) == '4' || s.at(i) == '7') // Check if the current digit is '4' or '7'.
      count++; // Increment count if it's a lucky digit.
  }
  
  // Functional Utility: Check if the `count` of lucky digits is itself a lucky number.
  // Postcondition: Prints "YES" if `count` is lucky, otherwise "NO".
  if (isLucky(count))
    printf("YES\n"); // Print "YES" if `count` is a lucky number.
  else
    printf("NO\n"); // Print "NO" if `count` is not a lucky number.
  return 0; // Indicate successful program execution.
}