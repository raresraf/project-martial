#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<ll> vi;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  if (n % 2 == 0 && n > 2)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
