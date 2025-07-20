/*
        SubmissionId	:	16012560
        ContestId	:	231
        Index	:	A
        ProblemName	:	Team
        ProblemTags	:	['brute force', 'greedy']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n;
  cin >> n;
  long long int x, i, j, p = 0, t = 0;
  for (i = 0; i < n; i++) {
    t = 0;
    for (j = 0; j < 3; j++) {
      cin >> x;
      if (x == 1)
        t += 1;
    }
    if (t > 1)
      p += 1;
  }
  cout << p;
  return 0;
}