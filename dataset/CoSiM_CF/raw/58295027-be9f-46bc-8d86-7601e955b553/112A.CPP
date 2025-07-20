//  [amitdu6ey]
#include <bits/stdc++.h>
#define bug(x) cout << "$ " << x << " $" << endl
#define ll long long
#define pb push_back
#define mp make_pair
#define tr(z) for (auto it = z.begin(); it != t.end(); it++)
#define loop(i, a, b) for (long long i = a; i < b; i++)
#define vbool vector<bool>
#define vvbool vector<vector<bool>>
#define vchar vector<char>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vll vector<long long>
#define vvl vector<vector<long long>>

using namespace std;

void solve() {
  string a, b;
  cin >> a >> b;
  int n = a.length();
  for (int i = 0; i < n; i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }
  for (int i = 0; i < n; i++) {
    if (a[i] - b[i] > 0) {
      cout << 1;
      return;
    } else if (a[i] - b[i] < 0) {
      cout << -1;
      return;
    }
  }
  cout << 0;
  return;
}

int main() {
  int test_cases = 1;
  // cin>>test_cases;
  while (test_cases--) {
    solve();
  }
  return 0;
}