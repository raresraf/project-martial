/*
        SubmissionId	:	15918774
        ContestId	:	282
        Index	:	A
        ProblemName	:	Bit++
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, i, j, x = 0;
  cin >> n;
  char a[1000];
  for (i = 0; i < n; i++) {
    cin >> a;
    if (a[0] == '+' || a[2] == '+')
      x += 1;
    else
      x -= 1;
  }
  cout << x;
  return 0;
}