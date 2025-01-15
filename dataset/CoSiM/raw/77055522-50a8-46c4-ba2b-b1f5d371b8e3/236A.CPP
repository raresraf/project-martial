/*
        SubmissionId	:	17429029
        ContestId	:	236
        Index	:	A
        ProblemName	:	Boy or Girl
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++11
        Verdict	:	OK
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
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  string a;
  cin >> a;
  long long int i, j, k, d = 0, l;
  l = a.size();
  for (i = 0; i < l; i++) {
    if (a[i] != '.') {
      for (j = i + 1; j < l; j++) {
        if (a[i] == a[j]) {
          a[j] = '.';
          d += 1;
        }
      }
    }
  }
  if ((l - d) % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
  return 0;
}
