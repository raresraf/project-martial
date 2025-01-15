#include <bits/stdc++.h>
#define endl "\n"
#define lli long long int
#define ulli unsigned long long int

using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  int n, k;
  cin >> n >> k;

  vector<int> a(n);
  int count = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0) {
      count++;
    } else {
      break;
    }
  }
  cout << count << endl;
}