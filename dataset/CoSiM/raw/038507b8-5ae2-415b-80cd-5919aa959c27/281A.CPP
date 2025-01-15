/*
        SubmissionId	:	16346983
        ContestId	:	281
        Index	:	A
        ProblemName	:	Word Capitalization
        ProblemTags	:	['strings']
        ProgrammingLanguage	:	GNU C++11
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  if (s[0] >= 97 && s[0] <= 122)
    s[0] -= 32;
  cout << s;
  return 0;
}