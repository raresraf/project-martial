/*
        SubmissionId	:	15988647
        ContestId	:	71
        Index	:	A
        ProblemName	:	Way Too Long Words
        ProblemTags	:	['strings']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, i, j, k;
  cin >> n;
  string a;
  for (i = 0; i < n; i++) {
    cin >> a;
    k = a.size();
    if (k > 10) {
      cout << a[0] << k - 2 << a[k - 1];
    } else
      cout << a;

    cout << "\n";
  }
  return 0;
}