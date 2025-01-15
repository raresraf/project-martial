#include <bits/stdc++.h>
using namespace std;
int main() {
  int a, c, d;
  string b;
  int cnt = 0;
  cin >> a >> b;
  for (int i = 0; i < a; i++) {
    if (b[i] == b[i + 1]) {
      cnt++;
    }
  }
  cout << cnt << endl;
}