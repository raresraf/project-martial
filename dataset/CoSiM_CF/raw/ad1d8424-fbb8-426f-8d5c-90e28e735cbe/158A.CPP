#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;

  cin >> n >> k;

  int a[n];

  for (int i = 0; i < n; i++)
    cin >> a[i];

  int s = 0;

  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0) {
      // cout<<a[i]<<"  "<<k<<endl;
      s++;
    }
  }

  cout << s << endl;
}
