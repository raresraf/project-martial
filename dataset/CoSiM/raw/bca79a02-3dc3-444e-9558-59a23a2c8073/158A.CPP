#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, ara[200], count = 0;
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> ara[i];
  }

  int l = ara[k - 1];

  for (int i = 0; i < n; i++) {
    if (ara[i] >= l && ara[i] != 0) {
      count++;
    }
  }

  cout << count << endl;
}
