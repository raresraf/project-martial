/*
        SubmissionId	:	16644299
        ContestId	:	112
        Index	:	A
        ProblemName	:	Petya and Strings
        ProblemTags	:	['implementation', 'strings']
        ProgrammingLanguage	:	GNU C++11
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int i, j, k, l;
  string a, b;
  cin >> a >> b;
  l = a.size();
  for (i = 0; i < l; i++) {
    if (a[i] >= 65 && a[i] <= 90)
      a[i] += 32;
    if (b[i] >= 65 && b[i] <= 90)
      b[i] += 32;
  }
  for (i = 0; i < l; i++) {
    if (a[i] > b[i]) {
      cout << "1";
      return 0;
    }
    if (a[i] < b[i]) {
      cout << "-1";
      return 0;
    }
  }
  cout << "0";
  return 0;
}