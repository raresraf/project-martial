#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pi;

int main() {

  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  cin >> n;

  bool res = n > 2 and n % 2 == 0;

  cout << (res ? "YES" : "NO") << "\n";

  return 0;
}
