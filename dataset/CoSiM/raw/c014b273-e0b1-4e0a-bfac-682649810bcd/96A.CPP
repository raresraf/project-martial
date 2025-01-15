/*
        SubmissionId	:	16268124
        ContestId	:	96
        Index	:	A
        ProblemName	:	Football
        ProblemTags	:	['implementation', 'strings']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  char a[101], b[] = "0000000", c[] = "1111111";
  cin >> a;
  if (strstr(a, b) || strstr(a, c)) {
    cout << "YES";
  } else
    cout << "NO";
  return 0;
}