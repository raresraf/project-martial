#include <bits/stdc++.h>
using namespace std;
int main() {
  string s1;
  int nmbr[105];
  int j = 0;
  cin >> s1;
  int len = s1.length();
  for (int i = 0; i < len; i++) {
    if (s1[i] != '+') {
      nmbr[j] = s1[i] - '0';
      j++;
    }
  }
  sort(nmbr, nmbr + j);
  for (int i = 0; i < j; i++) {
    cout << nmbr[i];
    if (i != j - 1) {
      cout << "+";
    }
  }
  cout << endl;
  return 0;
}
