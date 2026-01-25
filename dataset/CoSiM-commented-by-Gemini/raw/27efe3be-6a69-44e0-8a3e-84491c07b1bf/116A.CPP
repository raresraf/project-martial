/*
        SubmissionId	:	15918867
        ContestId	:	116
        Index	:	A
        ProblemName	:	Tram
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, i, j, c = 0, p = 0, a, b;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a >> b;
    p -= a;
    p += b;
    if (p > c)
      c = p;
  }
  cout << c;
  return 0;
}