#include <iostream>

using namespace std;

#define long long ll;

int n;
string s;

void solve() {
  int ngbrs = 0;
  for (int i = 0; i < n - 1; i++) {

    if (s[i] == s[i + 1])
      ngbrs++;
  }
  cout << ngbrs;
}

int main() {
  cin >> n;
  cin >> s;
  solve();
}