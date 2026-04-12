/**
 * @file 110A.CPP
 * @brief Logic for identifying "Nearly Lucky" numbers based on recursive digit classification.
 * 
 * Logic: A number is "nearly lucky" if the cardinality of its lucky digits 
 * (subset {4, 7}) is itself a "lucky number" (consisting only of digits {4, 7}).
 * 
 * Algorithm: 
 * 1. Single-pass digit extraction to count lucky digits.
 * 2. Secondary classification of the count itself.
 * 
 * Time Complexity: $O(\log_{10} N)$ where N is the input number.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <bits/stdc++.h>
using namespace std;

#define ll long long

/**
 * @brief Functional Utility: Determines if a numeric value is a "Lucky Number".
 * @param x The count of lucky digits to evaluate.
 * @return True if 'x' consists solely of digits 4 and 7, False otherwise.
 */
bool checkLucky(int x) {
  if (x == 0) return false;
  // Case 1: Single-digit lucky numbers.
  if (x < 10 && (x == 4 || x == 7)) {
    return true;
  } 
  // Case 2: Double-digit lucky numbers (44, 47, 74, 77).
  else if ((x >= 10 && x < 100) &&
             ((x / 10 == 4 || x / 10 == 7) &&
              (x % 10 == 4 || x % 10 == 7))) {
    return true;
  } 
  // Case 3: Triple-digit lucky numbers.
  else if ((x >= 100 && x < 1000) && 
             ((x % 10 == 7 || x % 10 == 4)) &&
             (x / 10 == 44 || x / 10 == 47 || x / 10 == 77 || x / 10 == 74)) {
    return true;
  } 
  return false;
}

int main() {
  ll n;
  // Reads the target number for analysis (up to 10^18).
  if (!(cin >> n)) return 0;
  
  ll temp = n;
  // Accumulator for the count of lucky digits.
  int lucky_count = 0;

  /**
   * Block Logic: Digit-by-digit extraction and filtering.
   * Invariant: Processes every digit of the input number exactly once.
   */
  while (temp) {
    int digit = temp % 10;
    // Logic: Identifies subset membership in {4, 7}.
    if (digit == 7 || digit == 4)
      lucky_count++;
    temp = temp / 10;
  }

  /**
   * Functional Utility: Final classification.
   * Logic: If the resulting count is a lucky number, the original input is "nearly lucky".
   */
  if (checkLucky(lucky_count))
    cout << "YES";
  else {
    cout << "NO";
  }

  return 0;
}

/**
 * Note: Following functions (mpow, ipgraph, dfs) are included in the source but 
 * are auxiliary and not utilized in the core "Nearly Lucky" logic.
 */
int mpow(int base, int exp) {
  const int mod = 1000000007;
  base %= mod;
  int result = 1;
  while (exp > 0) {
    if (exp & 1)
      result = ((ll)result * base) % mod;
    base = ((ll)base * base) % mod;
    exp >>= 1;
  }
  return result;
}
