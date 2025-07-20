/*
        SubmissionId	:	15987878
        ContestId	:	158
        Index	:	A
        ProblemName	:	Next Round
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, k, i, j, m = 0;
  cin >> n >> k;
  long long int a[n];
  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (j = n - 1; j >= 0; j--) {
    if (a[j] >= a[k - 1] && a[j] != 0)
      m += 1;
  }
  cout << m;
  return 0;
}