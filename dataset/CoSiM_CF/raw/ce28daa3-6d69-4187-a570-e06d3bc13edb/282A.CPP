#include <bits/stdc++.h>
#include <vector>
using namespace std;
int main() {
  int n;
  cin >> n;
  vector<string> v(n);
  for (int i = 0; i < n; i++)
    cin >> v[i];
  cout << count(v.begin(), v.end(), "++X") + count(v.begin(), v.end(), "X++") -
              count(v.begin(), v.end(), "--X") -
              count(v.begin(), v.end(), "X--")
       << endl;
  return 0;
}
// more neater version of count ... count(all(v)," ");
