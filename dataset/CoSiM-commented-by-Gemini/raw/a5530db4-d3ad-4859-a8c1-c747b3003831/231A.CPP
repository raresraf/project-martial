#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, c, a;
  cin >> t;
  int count = 0;
  for (int i = 0; i < t; ++i) {
    c = 0;
    for (int i = 0; i < 3; ++i) {
      cin >> a;
      if (a == 1)
        c++;
    }
    if (c >= 2)
      count++;
  }

  cout << count;
}