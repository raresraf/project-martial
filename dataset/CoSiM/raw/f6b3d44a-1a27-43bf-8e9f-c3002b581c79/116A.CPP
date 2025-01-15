#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int n;
  vector<int> c;
  int a[n], b[n];

  cin >> n;

  int count = 0;
  for (int i = 0; i < n; ++i) {
    cin >> a[i] >> b[i];
    count += -a[i] + b[i];
    c.push_back(count);
  }
  cout << *max_element(c.begin(), c.end());

  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }