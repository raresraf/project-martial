#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef unsigned long long ull;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);

  int T;
  cin >> T;
  while (T--) {
    int n;
    cin >> n;
    int c3 = 0, c5 = 0, c7 = 0;
    while (n > 0) {
      if (n % 7 == 0)
        c7 = n / 7, n = 0;
      else if (n % 5 == 0)
        c5 = n / 5, n = 0;
      else
        n -= 3, c3++;
    }

    if (n)
      cout << "-1\n";
    else
      cout << c3 << " " << c5 << " " << c7 << "\n";
  }
}
