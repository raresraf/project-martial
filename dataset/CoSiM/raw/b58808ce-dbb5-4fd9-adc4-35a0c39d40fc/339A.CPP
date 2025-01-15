#include <bits/stdc++.h>
using namespace std;

int main() {
  string inp;
  cin >> inp;
  int a[4] = {0};
  int len = inp.length();
  int plus_count = 0;
  for (int i = 0; i < len; i += 2) {
    int j = (int)inp[i] - 48;
    a[j]++;
    plus_count++;
  }

  for (int i = 1; i < 4; i++) {
    while (a[i] > 0) {
      cout << i;
      plus_count--;
      if (plus_count > 0)
        cout << '+';
      a[i]--;
    }
  }
}
