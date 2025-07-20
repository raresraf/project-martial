//  [amitdu6ey]
#include <bits/stdc++.h>
#define bug(x) cout << "$ " << x << " $" << endl
#define ll long long
#define ull unsigned long long
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
  ull n;
  int c = 0;
  cin >> n;
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7) {
      c++;
    }
    n /= 10;
  }
  if (c == 7 || c == 4) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
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