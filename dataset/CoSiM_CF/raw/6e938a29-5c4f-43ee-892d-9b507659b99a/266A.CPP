/*
        SubmissionId	:	20322619
        ContestId	:	266
        Index	:	A
        ProblemName	:	Stones on the Table
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++11
        Verdict	:	OK
*/

#include <bits/stdc++.h>
typedef long long ll;
typedef unsigned long long ull;
#define pi                                                                     \
  3.141592653589793238462643383279502884197169399375105820974944592307816406286
#define mod 1000000007
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

long long int gcd(long long int a, long long int b) {
  if (b == 0)
    return a;
  else
    return gcd(b, a % b);
}

long long int exp(long long int value, long long int power) {
  long long int result = 1;
  while (power > 0) {
    if (power & 1LL) {
      result *= value;
      result %= mod;
    }
    value *= value;
    value %= mod;
    power /= 2LL;
  }
  return result;
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  long long int f, cnt = 0, len, p, q, r, t, i, j, k, l, n, m, x, y, z, b, c, s;
  cin >> n;
  string a;
  cin >> a;
  for (i = 0; i < (n - 1); i++) {
    if (a[i] == a[i + 1])
      cnt += 1;
  }
  cout << cnt;

  return 0;
}
