#include <bits/stdc++.h>
#define lld long long int
#define vi vector<int>
#define vlld vector<lld>
#define pb push_back
#define ff first
#define ss second
#define pii pair<int, int>
#define mii map<int, int>
#define vii vector<pii>

using namespace std;

int main() {
  std::ios::sync_with_stdio(false);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n, x = 0;
  cin >> n;
  string s;
  while (n--) {
    cin >> s;
    (s[1] == '+') ? x++ : x--;
  }
  cout << x;
  return 0;
}