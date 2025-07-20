#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, total = 0, minec = 0, minev = 0;
  cin >> n;
  int ar[n];
  for (int i = 0; i < n; i++) {
    cin >> ar[i];
    total += ar[i];
  }
  sort(ar, ar + n);
  for (int i = n - 1; i >= 0; i--) {
    minev += ar[i];
    total -= ar[i];
    minec++;
    if (minev > total) {
      break;
    }
  }
  cout << minec << endl;
}
