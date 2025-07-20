#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, x = 0;
  cin >> n;
  string str;
  while (n--) {
    cin >> str;
    if (str == "++X" || str == "X++")
      ++x;
    else
      --x;
  }
  cout << x << endl;
}
