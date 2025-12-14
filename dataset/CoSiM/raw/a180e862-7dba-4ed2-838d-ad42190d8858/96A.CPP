#include <bits/stdc++.h>
#include <string>
using namespace std;

int isSubstring(string s1, string s2) {
  int M = s1.length();
  int N = s2.length();
  for (int i = 0; i <= N - M; i++) {
    int j;
    for (j = 0; j < M; j++)
      if (s2[i + j] != s1[j])
        break;
    if (j == M)
      return i;
  }
  return -1;
}

int main() {
  string s, s1 = "1111111", s0 = "0000000";
  getline(cin, s);
  int res1 = isSubstring(s1, s);
  int res0 = isSubstring(s0, s);
  if (res1 == -1 && res0 == -1)
    cout << "NO";
  else
    cout << "YES";
  return 0;
}
