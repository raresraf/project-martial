#include <bits/stdc++.h>

using namespace std;

int main() {
  int ct, i, j, a, b;
  ;
  vector<int> v[100];
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      int t;
      cin >> t;
      v[i].push_back(t);
    }
  }
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      if (v[i][j] == 1) {
        a = i;
        b = j;
      }
    }
  }
  ct = abs(a - 2) + abs(b - 2);
  cout << ct << endl;
}
