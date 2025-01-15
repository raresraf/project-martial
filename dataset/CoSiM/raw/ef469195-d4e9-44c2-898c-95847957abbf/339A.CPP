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
  for (i = 0; i < l; i += 2) {
    n[k] = a[i];
    k += 1;
  }
  sort(n, n + k);
  k = 0;
  for (i = 0; i < l; i += 2) {
    a[i] = n[k];
    k += 1;
  }
  cout << a;

  return 0;
}
