#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int pos = 0;
  for (int i = 1; i <= 100; i++) {
    int s = (i * (i + 1)) / 2;
    if ((n - s) > i) {
      pos = i;
    } else
      break;
  }
  int dif = n - (pos * (pos + 1)) / 2;
  cout << pos + 1 << endl;
  for (int i = 1; i < pos; i++) {
    cout << i << " ";
  }
  cout << dif;
}
