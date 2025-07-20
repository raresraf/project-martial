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
  string str;
  cin >> str;
  int curr = 1, highest = 1;
  for (int i = 1; i < str.size(); i++) {
    if (str[i] == str[i - 1]) {
      curr++;
      highest = max(highest, curr);
    } else {
      curr = 1;
    }
  }
  if (highest >= 7) {
    cout << "YES\n";
  } else
    cout << "NO\n";
}