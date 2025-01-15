#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, k;
  cin >> n >> k;
  int ar[n + 10], count = 0;

  for (int i = 1; i <= n; i++) {
    cin >> ar[i];
  }
  for (int i = 1; i <= n; i++) {
    if (ar[k] <= ar[i] && ar[i] > 0)
      count++;
  }
  cout << count << endl;
  return 0;
}
