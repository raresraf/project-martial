#include <algorithm>
#include <iostream>

using namespace std;

#define long long ll;

string s1, s2;

void solve() {
  int i = 0, j = 0;
  bool found = false;
  while (i < s1.length() && i < s2.length() && !found) {
    int c1 = int(s1[i]);
    int c2 = int(s2[i]);
    if (c1 != c2) {
      j = (c1 - c2) / abs(c1 - c2);
      found = true;
    }
    i++;
  }
  cout << j;
}

int main() {
  cin >> s1 >> s2;
  transform(s1.begin(), s1.end(), s1.begin(),
            [](unsigned char c) -> unsigned char { return tolower(c); });
  transform(s2.begin(), s2.end(), s2.begin(),
            [](unsigned char c) -> unsigned char { return tolower(c); });
  solve();
}