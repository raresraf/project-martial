/*
        SubmissionId	:	16012481
        ContestId	:	4
        Index	:	A
        ProblemName	:	Watermelon
        ProblemTags	:	['brute force', 'math']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int w, x = 0, i;
  cin >> w;
  if (w < 3)
    cout << "NO";
  else if (w % 2 == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}