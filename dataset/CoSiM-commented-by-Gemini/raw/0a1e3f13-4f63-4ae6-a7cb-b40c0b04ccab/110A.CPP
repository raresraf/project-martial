/**
 * @file 110A.CPP
 * @brief Determines if a number is a "Nearly Lucky Number".
 * @details This program solves the "Nearly Lucky Number" problem from Codeforces.
 * A number is "nearly lucky" if the count of its lucky digits (4 and 7) is itself a
 * lucky number. This implementation converts the number to a string to count its digits.
 *
 * @author Monil Soni sonil06
 * @school DA-IICT
 *
 * @see https://codeforces.com/problemset/problem/110/A
 *
 * Algorithm:
 * 1. The input number `n` is read and converted to its string representation.
 * 2. The program iterates through the string, counting the number of '4' and '7' characters.
 * 3. The final count is passed to a helper function `isLucky`.
 * 4. The `isLucky` function determines if the count is a lucky number (i.e., composed
 *    only of 4s and 7s). For the constraints of this problem, this effectively checks if
 *    the count is 4 or 7.
 * 5. The program prints "YES" or "NO" based on the result.
 *
 * Time Complexity: O(log10(N)), where N is the input number, as the complexity is
 * proportional to the number of digits in N.
 * Space Complexity: O(log10(N)) to store the string representation of N.
 */

// Includes a wide range of standard library headers, typical for competitive programming.
#include <bits/stdc++.h>

using namespace std;

// --- Boilerplate from a competitive programming template ---
typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;
#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
// ... other macros and template functions ...
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()
// --- End of boilerplate ---


/**
 * @brief Checks if a number consists only of the digits 4 and 7.
 * @param a The integer to check.
 * @return True if the number is lucky, false otherwise.
 * @note This implementation works by checking for the absence of any non-lucky digits.
 */
bool isLucky(int a) {
  string s = to_string(a);
  for(char c : s){
      if(c != '4' && c != '7')
          return false;
  }
  return true;
}

int main() {
  long long int n;
  cin >> n;
  int count = 0;
  
  // Convert the number to a string to easily iterate over its digits.
  string s = to_string(n);

  // Count the occurrences of lucky digits '4' and '7'.
  for (int i = 0; i < s.size(); i++) {
    if (s.at(i) == '4' || s.at(i) == '7')
      count++;
  }
  
  // A number is "nearly lucky" if the count of its lucky digits is a lucky number.
  // For this problem's constraints, this means the count must be 4 or 7.
  if (count == 4 || count == 7)
    printf("YES\n");
  else
    printf("NO\n");
    
  return 0;
}
