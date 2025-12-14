#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, x = 0;
  string word;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> word;
    if (word == "++X" || word == "X++")
      x++;
    else
      x--;
  }

  cout << x << endl;

  return 0;
}