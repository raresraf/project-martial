#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int n, k, c, p;
  cin >> n >> k;

  int arr[n];
  c = 0;

  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  p = arr[k - 1];

  for (int i = 0; i < n; ++i) {
    if (arr[i] > 0 && arr[i] >= p) {
      ++c;
    }
  }
  cout << c;
  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }