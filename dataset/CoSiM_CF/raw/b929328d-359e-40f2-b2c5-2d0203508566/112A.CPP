#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string a, b;
  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  cout << a.compare(b);
}
