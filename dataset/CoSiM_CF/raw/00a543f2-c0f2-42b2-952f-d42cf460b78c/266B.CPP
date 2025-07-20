#include <iostream>

using namespace std;

#define ll long long;

int n, t;
string s;

void solve() {
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        j++;
      }
    }
  }
  cout << s;
}

int main() {
  cin >> n >> t >> s;
  solve();
}