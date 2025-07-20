#include <bits/stdc++.h>
using namespace std;

int main() {
  char s[200];
  cin >> s;
  int l = strlen(s);
  int a = l - (l / 2);
  int j = 0;
  int t[200];
  for (int i = 0; i < a; i++) {
    t[i] = s[j];
    j = j + 2;
  }
  sort(t, t + a);
  for (int k = 0; k < a; k++) {
    cout << t[k] - 48;
    if (k < (a - 1))
      cout << "+";
  }
  return 0;
}
