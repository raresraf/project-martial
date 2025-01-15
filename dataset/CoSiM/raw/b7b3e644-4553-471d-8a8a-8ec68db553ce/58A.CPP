/*
        SubmissionId	:	16288964
        ContestId	:	58
        Index	:	A
        ProblemName	:	Chat room
        ProblemTags	:	['greedy', 'strings']
        ProgrammingLanguage	:	GNU C++11
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;
int main() {
  char a[10000];
  cin >> a;
  int i, k = 0;
  for (i = 0; a[i]; i++) {
    if (a[i] == 'h') {
      k = 1;
      break;
    }
  }
  if (k == 1) {
    for (i = i + 1; a[i]; i++) {
      if (a[i] == 'e') {
        k = 2;
        break;
      }
    }
  }
  if (k == 2) {
    for (i = i + 1; a[i]; i++) {
      if (a[i] == 'l') {
        k = 3;
        break;
      }
    }
  }
  if (k == 3) {
    for (i = i + 1; a[i]; i++) {
      if (a[i] == 'l') {
        k = 4;
        break;
      }
    }
  }
  if (k == 4) {
    for (i = i + 1; a[i]; i++) {
      if (a[i] == 'o') {
        k = 5;
        break;
      }
    }
  }
  if (k == 5)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}