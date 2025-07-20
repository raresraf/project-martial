#include <bits/stdc++.h>
using namespace std;
int main() {
  int N;
  cin >> N;
  int x = 0;
  string stat;
  while (N--) {
    cin >> stat;
    if (stat[1] == '-')
      x = x - 1;
    else
      x = x + 1;
  }
  cout << x << endl;
}
