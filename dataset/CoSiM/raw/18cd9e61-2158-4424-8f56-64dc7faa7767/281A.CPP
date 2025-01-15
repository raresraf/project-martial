#include <algorithm>
#include <iostream>

using namespace std;

#define long long ll;

string s;

void solve() {
  transform(s.begin(), s.begin() + 1, s.begin(),
            [](unsigned char c) -> unsigned char { return toupper(c); });
  cout << s;
}

int main() {
  cin >> s;

  solve();
}