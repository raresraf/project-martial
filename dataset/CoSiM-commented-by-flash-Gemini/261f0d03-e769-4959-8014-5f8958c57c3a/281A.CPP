#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  string s;
  cin >> s;
  if (s[0] > 'Z')
    s[0] -= 32;
  cout << s << "\n";
}
