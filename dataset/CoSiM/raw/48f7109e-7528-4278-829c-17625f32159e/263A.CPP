
#include <bits/stdc++.h>

using namespace std;

string a, b, c;

int main() {
  cin >> a >> b;
  int n = (int)(a.size());
  for (int i = 0; i < n; i++)
    c.push_back(char((a[i] != b[i]) + '0'));
  cout << c;
}
