#include <bits/stdc++.h>
using namespace std;
int a[3];

int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i += 2) {
    if (s[i] == '1')
      a[0]++;
    else if (s[i] == '2')
      a[1]++;
    else
      a[2]++;
  }

  if (a[0] > 0) {
    cout << '1';
    a[0]--;
  } else if (a[1] > 0) {
    cout << '2';
    a[1]--;
  } else if (a[2] > 0) {
    cout << '3';
    a[2]--;
  }

  for (int j = 0; j < 3; j++)
    for (int i = 1; i <= a[j]; i++)
      cout << '+' << j + 1;
  return 0;
}