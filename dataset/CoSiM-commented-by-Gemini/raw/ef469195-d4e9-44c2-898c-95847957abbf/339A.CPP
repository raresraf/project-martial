/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details This program takes a string representing a sum of single-digit numbers
 * and rearranges the numbers into non-decreasing order.
 *
 * @par Algorithm
 * 1. Read the input string representing the sum (e.g., "3+2+1").
 * 2. Extract the numbers, which are at even indices, and store them in an array.
 * 3. Sort the array of numbers.
 * 4. Place the sorted numbers back into the original string at the even indices.
 * 5. Print the modified string.
 *
 * @par Time Complexity
 * O(L + N log N), where L is the length of the string and N is the number of
 * digits. L is for iterating through the string, and N log N is for sorting.
 *
 * @par Space Complexity
 * O(N) to store the numbers in an auxiliary array.
 */
/*
        SubmissionId	:	17301688
        ContestId	:	339
        Index	:	A
        ProblemName	:	Helpful Maths
        ProblemTags	:	['greedy', 'implementation', 'sortings',
   'strings'] ProgrammingLanguage	:	GNU C++11 Verdict	:
   OK
*/

#include <bits/stdc++.h>
typedef long long ll;
typedef unsigned long long ull;
#define pi                                                                     \
  3.141592653589793238462643383279502884197169399375105820974944592307816406286
using namespace std;
/**
 * @brief Checks if a number is prime. This function is not used in the solution.
 * @param n The number to check.
 * @return True if n is prime, false otherwise.
 */
bool isprime(ull n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (ull i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return false;
  return true;
}
int main() {
  // ios_base::sync_with_stdio(0);cin.tie(NULL);cout.tie(NULL);
  string a;
  cin >> a;
  long long int l, i, j, k, n[101];
  l = a.size();
  k = 0;
  // Extract the numbers from the string.
  for (i = 0; i < l; i += 2) {
    n[k] = a[i];
    k += 1;
  }
  // Sort the array of numbers.
  sort(n, n + k);
  k = 0;
  // Place the sorted numbers back into the string.
  for (i = 0; i < l; i += 2) {
    a[i] = n[k];
    k += 1;
  }
  cout << a;

  return 0;
}
