#include <bits/stdc++.h>
#define ll long long int
#define pb push_back

using namespace std;

int main() {
  int n, a, b, t = 0, h = -1;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;

    t = t - a;
    t = t + b;

    if (t > h)
      h = t;
  }

  cout << h << endl;
}
