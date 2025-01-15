#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, tot = 0;
  cin >> n >> k;
  int ar[n];
  for (int i = 0; i < n; i++) {
    cin >> ar[i];
  }
  for (int i = 0; i < n; i++) {
    if (ar[i] > 0 and ar[i] >= ar[k - 1]) {
      tot++;
    }
  }
  cout << tot << "\n";
}
