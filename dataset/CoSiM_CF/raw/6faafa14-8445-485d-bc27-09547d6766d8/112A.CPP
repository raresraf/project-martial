#include <bits/stdc++.h>
using namespace std;

int main() {
  char s1[100];
  cin >> s1;
  char s2[100];
  cin >> s2;
  int n = strlen(s1);
  int i = 0;
  for (i = 0; i < n; i++) {
    int a = 0;
    int b = 0;
    if (s1[i] < 97) {
      a = s1[i];
      a += 32;
    } else
      a = s1[i];

    if (s2[i] < 97) {
      b = s2[i];
      b += 32;
    } else
      b = s2[i];
    if (a < b) {
      cout << -1 << endl;
      break;
    } else if (a > b) {
      cout << 1 << endl;
      break;
    }
  }
  if (i == n)
    cout << 0 << endl;

  return 0;
}
