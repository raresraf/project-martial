/*
        SubmissionId	:	16346932
        ContestId	:	110
        Index	:	A
        ProblemName	:	Nearly Lucky Number
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++11
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, i, j, k = 0;
  cin >> n;
  while (n > 0) {
    if ((n % 10) == 7 || (n % 10) == 4)
      k += 1;
    n /= 10;
  }
  if (k == 7 || k == 4)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}