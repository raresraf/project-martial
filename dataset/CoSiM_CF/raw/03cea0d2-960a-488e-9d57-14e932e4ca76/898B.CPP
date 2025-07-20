#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef std::vector<int> vi;

int main(int argc, char const *argv[]) {
  ll n, a, b;
  cin >> n >> a >> b;
  if (n % a == 0) {
    cout << "YES" << endl << n / a << ' ' << '0';
    return 0;
  } else if (n % b == 0) {
    cout << "YES" << endl << '0' << ' ' << n / b;
    return 0;
  } else {
    ll tmp1 = n / a;
    for (int i = 1; i <= tmp1; i++) {
      if ((n - (i * a)) % b == 0) {
        cout << "YES" << endl << i << ' ' << (n - (i * a)) / b;
        return 0;
      }
    }
  }

  cout << "NO";
  return 0;
}
