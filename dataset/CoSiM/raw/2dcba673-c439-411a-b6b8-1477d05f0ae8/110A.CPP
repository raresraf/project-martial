#include <bits/stdc++.h>
using namespace std;
#define length(a) (int)a.size()
#define all(a) a.begin(), a.end()
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define lsb(x) (x & (-x))
#define ctoi(a) (int)a - 48
#define itoc(a) a + '0'
#define MOD 1000000007
#define ll long long

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  ll n;
  cin >> n;
  int count = 0;
  while (n != 0) {
    int a = n % 10;
    count += (a == 7 || a == 4);
    n /= 10;
  }
  // cout<<count<<"\n";
  int flag = !(count == 0);
  while (count != 0) {
    int a = count % 10;
    if (a != 7 && a != 4)
      flag = false;
    count /= 10;
  }
  if (flag)
    cout << "YES\n";
  else
    cout << "NO\n";
}